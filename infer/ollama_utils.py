from datetime import datetime
import ollama
import re

def get_filename(reply_text):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Remove any characters that are not safe for filenames
    filename_safe_text = re.sub(r'[<>:"/\\|?*]', '', reply_text.encode('ascii', 'ignore').decode())
    filename_safe_text = filename_safe_text[:150 - len(timestamp) - 5].rstrip(".")
    filename_safe_text = f"{timestamp}_{filename_safe_text}.wav"

    return filename_safe_text

def get_lyrics_prompt(prompt: str):
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
    Don't include info about anything else like, intro, outro, chorus, verse, bridges etc.
    You MUST follow the previous instructions about the format of the lyrics!.
    Please don't include anything extra, like repeating the prompt or anything else, just the lyrics!
    Also don't include any extra spacing and such.
    """
    #print(ollama_prompt)
    ollama_prompt += "The song should be about " + prompt
    #print(ollama_prompt)
    ollama_model = "deepseek-r1:14b"

    ollama_client = ollama.Client(host="http://localhost:11434")

    response = ollama_client.generate(model=ollama_model, prompt=ollama_prompt)

    reply_text = response['response'].strip()  # Adjust key if needed

    reply_text = re.sub(r"<think>.*?</think>", "", reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r"\"", "", reply_text, flags=re.DOTALL).strip()
    #print(reply_text)

    return reply_text

def get_musicstyle_prompt(prompt: str):
    ollama_prompt = """
    Hi! Today we are going to generate short music description for an AI to generate music.
    Here is an example output : "Futuristic, low key piano with an organ"
    Please reply in a similar manner, be brief." \
    """
    ollama_model = "deepseek-r1:14b"
    #print(ollama_prompt)

    ollama_prompt += "Use the following input to write a short sentance about the style: " + prompt
    ollama_client = ollama.Client(host="http://localhost:11434")

    response = ollama_client.generate(model=ollama_model, prompt=ollama_prompt)

    reply_text = response['response'].strip()  # Adjust key if needed

    reply_text = re.sub(r"<think>.*?</think>", "", reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r"\"", "", reply_text, flags=re.DOTALL).strip()
    #print(reply_text)

    return reply_text