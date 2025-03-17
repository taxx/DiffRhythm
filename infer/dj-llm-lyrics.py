# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Stability AI License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import torch
import torchaudio
from einops import rearrange

from datetime import datetime
import ollama
import re

#print("Current working directory:", os.getcwd())

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    chunked=False,
):
    with torch.inference_mode():
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
        )

        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]

        output = decode_audio(latent, vae_model, chunked=chunked)

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        return output

def get_lyrics_prompt():
    ollama_prompt = """
    Can you help me create lyrics for a song, here are the technical constraints:
    "Lyrics Format Requirements

        Each line must follow: [mm:ss.xx]Lyric content
        Example of valid format:
        "
        [00:10.00]You... only... you
        [00:23.20]You are ... my sunshine
        [00:43.20]You are ... my darkness
        [00:53.20]You are ... everything
        "
        Maximum 95 seconds songs
        Total timestamps should not exceed 01:35.00 (95 seconds)
    "

    The song could be about anything, but prefer pop, upbeat, fast paced.
    Don't include info about anything else like, intro, outro, chorus etc.
    Just keep the pure lyrics in the format described earlier.
    Don't include any extra spacing and such.
    """
    #print(ollama_prompt)
    ollama_model = "deepseek-r1:14b"

    ollama_client = ollama.Client(host="http://localhost:11434")

    response = ollama_client.generate(model=ollama_model, prompt=ollama_prompt)

    reply_text = response['response'].strip()  # Adjust key if needed

    reply_text = re.sub(r"<think>.*?</think>", "", reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r"\"", "", reply_text, flags=re.DOTALL).strip()
    #print(reply_text)

    return reply_text

def get_musicstyle_prompt():
    ollama_prompt = """
    Hi! Today we are going to generate short music description for an AI to generate music.
    Here is an example: 'Modern pop, female vocalist'.
    Please reply in a similar manner, be brief." \
    """
    ollama_model = "deepseek-r1:14b"
    #print(ollama_prompt)
    ollama_client = ollama.Client(host="http://localhost:11434")

    response = ollama_client.generate(model=ollama_model, prompt=ollama_prompt)

    reply_text = response['response'].strip()  # Adjust key if needed

    reply_text = re.sub(r"<think>.*?</think>", "", reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r"\"", "", reply_text, flags=re.DOTALL).strip()
    #print(reply_text)

    return reply_text

def get_filename(reply_text):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Remove any characters that are not safe for filenames
    filename_safe_text = re.sub(r'[<>:"/\\|?*]', '', reply_text)
    filename_safe_text = filename_safe_text[:150 - len(timestamp) - 5].rstrip(".")
    filename_safe_text = f"{timestamp}_{filename_safe_text}.wav"

    return filename_safe_text

if __name__ == "__main__":
    assert torch.cuda.is_available(), "only available on gpu"

    device = "cuda"

    chunked=True
    output_dir = "generated/songs"

    audio_length = 95
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:  # current not available
        max_frames = 6144

    cfm, tokenizer, muq, vae = prepare_model(device)

    # Prompt och filnamn
    #
    max_iterations = 5
    for i in range(max_iterations):
        # lyrics (tomt för instrumelta låtar)
        lrc = get_lyrics_prompt()
        lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)
        
        ref_prompt = get_musicstyle_prompt()
        output_filename = get_filename(ref_prompt)

        print(f"Iteration: {i} of {max_iterations}")
        print(ref_prompt)
        print(lrc)
        print(output_filename)

        style_prompt = get_style_prompt(muq, prompt=ref_prompt)

        negative_style_prompt = get_negative_style_prompt(device)

        latent_prompt = get_reference_latent(device, max_frames)

        s_t = time.time()
        generated_song = inference(
            cfm_model=cfm,
            vae_model=vae,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=max_frames,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            chunked=chunked,
        )
        e_t = time.time() - s_t
        print(f"inference cost {e_t} seconds")

        output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, output_filename)
        torchaudio.save(output_path, generated_song, sample_rate=44100)

        # Save lyrics to a text file
        lyrics_path = os.path.join(output_dir, output_filename.replace(".wav", ".txt"))
        with open(lyrics_path, "w", encoding="utf-8") as f:
            f.write(lrc)

        print("Sleeping for 30 seconds...")
        time.sleep(30)

