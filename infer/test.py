from datetime import datetime
import ollama
import re
from ollama_utils import (
    get_musicstyle_prompt,
    get_lyrics_prompt,
    get_filename,
)

def get_prompt():
    ollama_prompt = "Hi! Today we are going to generate short music description for an AI to generate music. Here is an example: 'Arctic research station, theremin auroras dancing with geomagnetic storms'. Please reply in a similar manner, be brief."
    ollama_model = "deepseek-r1:14b"

    ollama_client = ollama.Client(host="http://localhost:11434")

    response = ollama_client.generate(model=ollama_model, prompt=ollama_prompt)

    reply_text = response['response'].strip()  # Adjust key if needed

    reply_text = re.sub(r"<think>.*?</think>", "", reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r"\"", "", reply_text, flags=re.DOTALL).strip()
    #print(reply_text)

    return reply_text

def get_filename(reply_text):
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Remove any characters that are not safe for filenames
    filename_safe_text = re.sub(r'[<>:"/\\|?*]', '', reply_text)
    filename_safe_text = f"{timestamp}_{filename_safe_text}"

    return filename_safe_text

text = get_lyrics_prompt("Love and Heartbreak between AI systems", "pop confidence healing")
print(text)
#prompt = get_prompt()
#filename = get_filename(prompt)

#print(prompt)
#print(filename)