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

def get_lyrics_prompt(theme, tags_gen):
    ollama_prompt = """
    Please generate complete lyrics that align with the `{tags}` style, centered around the theme of "{theme}". Strictly adhere to the following requirements:  

    ### **Mandatory Formatting Rules**  
    1. **Only output timestamps and lyrics**; brackets, narration, and section labels (such as chorus, interlude, outro, etc.) are strictly prohibited.  
    2. Each line must follow the format `[mm:ss.xx]LyricsContent`, with no spaces between the timestamp and the lyrics. The lyrics must be complete and coherent.  
    3. Timestamps should be naturally distributed. **The first lyric must not start at [00:00.00]**—the intro silence must be taken into account.  

    ### **Content and Structure Requirements**  
    1. The lyrics should be rich in variation, with emotional progression and an overall sense of coherence and layering. **Each line's length should vary naturally**—do not make them uniform, as that would appear overly formulaic.  
    2. **Timestamps should be assigned based on the song's tags, emotional tone, and rhythm**, rather than being mechanically distributed based on lyric length.  
    3. Interludes and outros should be indicated solely by time gaps (e.g., jumping directly from `[02:30.00]` to `[02:50.00]`), **without any textual descriptions**.  

    Don't include info about anything else like, intro, outro, chorus, verse, bridges etc.
    You MUST follow the previous instructions about the format of the lyrics!.
    Please don't include anything extra, like repeating the prompt or anything else, just the lyrics!
    Also don't include any extra spacing and such.

    ### **Negative Examples (Prohibited)**  
    - Incorrect: `[01:30.00](Piano Interlude)`  
    - Incorrect: `[02:00.00][Chorus]`  
    - Incorrect: Blank lines, line breaks, or annotations
    - Incorrect: [01:45.00]
    """

    ollama_model = "deepseek-r1:14b"

    ollama_client = ollama.Client(host="http://localhost:11434")

    response = ollama_client.generate(
        model=ollama_model,
        prompt=ollama_prompt.format(theme=theme, tags=tags_gen),
        system="You are a professional musician who has been invited to make music-related comments.",
        stream=False,
    )

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