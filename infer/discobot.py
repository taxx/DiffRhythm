import asyncio
import os
import time
import discord
import random
from dotenv import load_dotenv

import torch
import torchaudio
from einops import rearrange

from discord.ext import commands
from discord import app_commands

from ollama_utils import (
    get_musicstyle_prompt,
    get_lyrics_prompt,
    get_filename,
)

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)

# Load environment variables from .env file
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
MUSIC_FOLDER = os.getenv("MUSIC_FOLDER")
output_dir = "/home/tobbe/source/DiffRhythm/generated"

intents = discord.Intents.default()
intents.message_content = True  # Needed for commands
intents.voice_states = True  # Needed for joining/leaving VC

bot = commands.Bot(command_prefix="!", intents=intents)
music_queue = []
volume_level = 0.3

#print(get_musicstyle_prompt("futuristic, apocalypse"))
#print(get_lyrics_prompt("random robot"))

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


def generate_music(music_style_prompt: str, lyrics_text: str):
    device = "cuda"

    chunked=True
    
    audio_length = 95
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:  # current not available
        max_frames = 6144

    cfm, tokenizer, muq, vae = prepare_model(device)

    ref_prompt = get_musicstyle_prompt(music_style_prompt)
    output_filename = get_filename(ref_prompt)

    # lyrics (tomt för instrumelta låtar)
    ##lrc = ""
    lrc_prompt, start_time = get_lrc_token(lyrics_text, tokenizer, device)

    #print(ref_prompt)
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

    #output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    torchaudio.save(output_path, generated_song, sample_rate=44100)

    return output_filename

def load_music():
    global music_queue
    music_queue = [f for f in os.listdir(MUSIC_FOLDER) if f.endswith(".wav")]
    random.shuffle(music_queue)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    load_music()
    await bot.tree.sync()  # Syncs slash commands with Discord
    print(bot.tree)
    # Print out all registered slash commands
    for command in await bot.tree.fetch_commands():
        print(f"Command: {command.name}, Description: {command.description}")


@bot.tree.command(name="lyrics", description="Get lyrics")
async def lyrics(interaction: discord.Interaction, subject: str):
    # Send an initial quick response
    await interaction.response.defer(ephemeral=True)

    # Run the long task (fetching lyrics) in a separate thread
    lyrics_text = await asyncio.to_thread(get_lyrics_prompt, subject)

    # Send the result after processing
    if lyrics_text:
        await interaction.followup.send(lyrics_text, ephemeral=True)  # Use followup here
    else:
        await interaction.followup.send("Sorry, I couldn't find the lyrics.", ephemeral=True)  # Use followup here


@bot.tree.command(name="generateandplay", description="Generate music and play it")
@app_commands.describe(
    music_style="The style of music you want to generate",
    lyrics_subject="[Optional] The subject of the lyrics"
)
async def generateandplay(interaction: discord.Interaction, music_style: str, lyrics_subject: str=""):
    if not music_style:
        await interaction.response.send_message("Music style is required!", ephemeral=True)
        return

    if interaction.user.voice and not interaction.guild.voice_client:
        channel = interaction.user.voice.channel
        await channel.connect()

    # Send an initial quick response
    await interaction.response.defer(ephemeral=True)

    lyrics_text = ""
    if lyrics_subject and lyrics_subject != "":
        lyrics_text = await asyncio.to_thread(get_lyrics_prompt, lyrics_subject)
        if len(lyrics_text) > 2000:
            chunks = [lyrics_text[i:i+2000] for i in range(0, len(lyrics_text), 2000)]
            for chunk in chunks:
                await interaction.followup.send(f"Lyrics: \r\n {chunk}", ephemeral=True)
        else:
            await interaction.followup.send(f"Lyrics: \r\n {lyrics_text}", ephemeral=True)

    # Run the long task (fetching lyrics) in a separate thread
    music_style_prompt = await asyncio.to_thread(get_musicstyle_prompt, music_style)
    await interaction.followup.send("Creating " + music_style_prompt + ". Please wait...", ephemeral=True)

    generated_file_path = await asyncio.to_thread(generate_music, music_style_prompt, lyrics_text)

    # Send the result after processing
    if generated_file_path:
        print(f"Generated file: {generated_file_path}")
        print(f"Path {output_dir}")

        if interaction.guild.voice_client and interaction.guild.voice_client.is_playing():
            interaction.guild.voice_client.stop()

        source = discord.FFmpegPCMAudio(os.path.join(output_dir, generated_file_path))
        volume_source = discord.PCMVolumeTransformer(source, volume_level)
        interaction.guild.voice_client.play(volume_source, after=lambda e: bot.loop.create_task(play_next(interaction)))
        await interaction.followup.send(f"Playing `{generated_file_path}` at volume {volume_level}!", ephemeral=True)

    else:
        await interaction.followup.send("Sorry, I couldn't generate the music.", ephemeral=True)  # Use followup here


@bot.event
async def on_message(message):
    # don't respond to ourselves
    if message.content == 'ping':
        await message.channel.send('pong')


@bot.tree.command(name="join", description="Bot joins your voice channel")
async def join(interaction: discord.Interaction):
    if interaction.user.voice:
        channel = interaction.user.voice.channel
        await channel.connect()
        await interaction.response.send_message(f"Joined {channel.name}!", ephemeral=True)
    else:
        await interaction.response.send_message("You need to be in a voice channel!", ephemeral=True)


@bot.tree.command(name="playrandom", description="Plays a shuffled WAV file from the folder")
async def playrandom(interaction: discord.Interaction, volume: float = volume_level):
    if interaction.guild.voice_client:
        if not music_queue:
            load_music()
        if music_queue:
            filename = music_queue.pop(0)
            source = discord.FFmpegPCMAudio(os.path.join(MUSIC_FOLDER, filename))
            volume_source = discord.PCMVolumeTransformer(source, volume)
            interaction.guild.voice_client.play(volume_source, after=lambda e: bot.loop.create_task(play_next(interaction)))
            await interaction.response.send_message(f"Playing `{filename}` at volume {volume}!", ephemeral=True)
        else:
            await interaction.response.send_message("No music files found!", ephemeral=True)
    else:
        await interaction.response.send_message("Bot is not in a voice channel! Use `/join` first.", ephemeral=True)


@bot.tree.command(name="playnext", description="Skips the current song")
async def play_next(interaction):
    if music_queue:
        filename = music_queue.pop(0)
        source = discord.FFmpegPCMAudio(os.path.join(MUSIC_FOLDER, filename))
        volume_source = discord.PCMVolumeTransformer(source)
        interaction.guild.voice_client.play(volume_source, after=lambda e: bot.loop.create_task(play_next(interaction)))


@bot.tree.command(name="stop", description="Stops the current music playback")
async def stop(interaction: discord.Interaction):
    if interaction.guild.voice_client and interaction.guild.voice_client.is_playing():
        interaction.guild.voice_client.stop()
        await interaction.response.send_message("Music stopped!", ephemeral=True)
    else:
        await interaction.response.send_message("No music is playing!", ephemeral=True)


@bot.tree.command(name="skip", description="Skips the current song")
async def skip(interaction: discord.Interaction):
    if interaction.guild.voice_client and interaction.guild.voice_client.is_playing():
        interaction.guild.voice_client.stop()
        await interaction.response.send_message("Skipping to the next song...", ephemeral=True)
        await play_next(interaction)
    else:
        await interaction.response.send_message("No music is playing!", ephemeral=True)


@bot.tree.command(name="volume", description="Adjusts the playback volume")
async def set_volume(interaction: discord.Interaction, level: float):
    if interaction.guild.voice_client and interaction.guild.voice_client.source:
        if 0.0 <= level <= 2.0:
            interaction.guild.voice_client.source.volume = level
            volume_level = level
            await interaction.response.send_message(f"Volume set to {level}!", ephemeral=True)
        else:
            await interaction.response.send_message("Volume must be between 0.0 and 2.0!", ephemeral=True)
    else:
        await interaction.response.send_message("No audio is playing!")


@bot.tree.command(name="leave", description="Bot leaves the voice channel")
async def leave(interaction: discord.Interaction):
    if interaction.guild.voice_client:
        await interaction.guild.voice_client.disconnect()
        await interaction.response.send_message("Left the voice channel!", ephemeral=True)
    else:
        await interaction.response.send_message("I'm not in a voice channel!", ephemeral=True)


@bot.command()
async def sync(ctx):
    await bot.tree.sync()
    await ctx.send("Commands synced!")


bot.run(TOKEN)
