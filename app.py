import streamlit as st
import moviepy.editor as mp
import os
import numpy as np
import tempfile
from groq import Groq
import assemblyai as aai
from gtts import gTTS
import json
from pydub import AudioSegment
import soundfile as sf  
from dotenv import load_dotenv
import imageio_ffmpeg as ffmpeg
import shutil
import subprocess

# Load environment variables from .env file
load_dotenv()

# Check if ffmpeg is available
def check_ffmpeg_installation():
    try:
        # Get paths to ffmpeg and ffprobe from imageio_ffmpeg
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        ffprobe_exe = ffmpeg.get_ffprobe_exe()
        
        # Check if FFmpeg is available
        subprocess.run([ffmpeg_exe, "-version"], check=True, stdout=subprocess.PIPE)
        print("FFmpeg is available.")

        # Check if FFprobe is available
        subprocess.run([ffprobe_exe, "-version"], check=True, stdout=subprocess.PIPE)
        print("FFprobe is available.")

        st.success("FFmpeg and FFprobe installation successful.")
    except subprocess.CalledProcessError as e:
        st.error("FFmpeg or FFprobe not found.")
        st.stop()  # Stop execution if ffmpeg or ffprobe is not available

def check_ffmpeg_and_ffprobe():
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    ffprobe_exe = shutil.which("ffprobe")  # Using shutil to check if ffprobe exists in the PATH
    if not ffprobe_exe:
        st.error("FFprobe not found in the system path. Please install ffprobe or add it to the system path.")
        return False
    print(f"FFmpeg: {ffmpeg_exe} is available!")
    print(f"FFprobe: {ffprobe_exe} is available!")
    return True

if not check_ffmpeg_and_ffprobe():
    st.stop()

aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(speaker_labels=True, word_boost=["um", "hmm"], boost_param='high')

# Function to transcribe audio using AssemblyAI
def transcribe_audio_with_timestamps(audio_file):
    transcript = transcriber.transcribe(audio_file, config)
    if transcript.status == aai.TranscriptStatus.error:
        st.write(f"Transcription failed: {transcript.error}")
        return None
    else:
        st.write("Transcription successful")
        return transcript

# Function to correct transcription using GROQ_API_KEY
def correct_transcription(transcription):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Correct the transcription below by removing unnecessary words like 'um', 'hmm', and also DO NOT REMOVE EXTRA WORDS OR SENTENCES and give the result without adding any extra commentary."},
            {"role": "user", "content": transcription}
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

# Function to convert text to speech and align it with word timestamps
def synthesize_speech_with_timeline(corrected_transcription, word_timestamps):
    max_chunk_size = 5000  
    chunks = []
    while len(corrected_transcription) > max_chunk_size:
        split_index = corrected_transcription[:max_chunk_size].rfind('.')
        if split_index == -1: 
            split_index = max_chunk_size

        chunks.append(corrected_transcription[:split_index + 1].strip()) 
        corrected_transcription = corrected_transcription[split_index + 1:] 

    chunks.append(corrected_transcription.strip())  
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio_file.close()  
    for i, chunk in enumerate(chunks):
        tts = gTTS(text=chunk, lang='en')
        temp_chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_chunk_file.close()

        tts.save(temp_chunk_file.name)
        if i == 0:
            os.rename(temp_chunk_file.name, temp_audio_file.name)  
        else:
            os.system(f"ffmpeg -y -i {temp_audio_file.name} -i {temp_chunk_file.name} -filter_complex '[0:a][1:a]concat=n=2:v=0:a=1[out]' -map '[out]' {temp_audio_file.name}")
        if os.path.exists(temp_chunk_file.name):
            os.remove(temp_chunk_file.name)

    return temp_audio_file.name  # Return the path to the final audio file

# Function to map the corrected transcription to word timestamps
def map_corrected_transcription_to_timestamps(original_transcription, corrected_transcription, word_timestamps):
    original_words = original_transcription.split()
    corrected_words = corrected_transcription.split()
    aligned_timestamps = []

    original_idx, corrected_idx = 0, 0

    while original_idx < len(original_words) and corrected_idx < len(corrected_words):
        if original_words[original_idx] == corrected_words[corrected_idx]:
            aligned_timestamps.append(word_timestamps[original_idx])
            corrected_idx += 1
        original_idx += 1

    return aligned_timestamps

# Function to increase the speed of the audio using Pydub
def increase_audio_speed_pydub(audio_file, speed_factor=1.2):
    audio = AudioSegment.from_file(audio_file)
    sped_up_audio = audio.speedup(playback_speed=speed_factor)
    temp_sped_up_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_sped_up_audio_file.close()
    sped_up_audio.export(temp_sped_up_audio_file.name, format="mp3")
    return temp_sped_up_audio_file.name 

# Function to replace audio in video with the newly generated audio
def replace_audio_in_video_with_timeline(video_file, new_audio, word_timestamps, speed_factor=1.12):
    video_clip = mp.VideoFileClip(video_file)
    new_audio_sped_up = increase_audio_speed_pydub(new_audio, speed_factor)
    new_audio_clip = mp.AudioFileClip(new_audio_sped_up)
    new_audio_duration = new_audio_clip.duration
    original_audio_duration = video_clip.audio.duration
    if new_audio_duration < original_audio_duration:
        # If the audio is shorter, add silence to the audio to match the video duration
        silence_duration = original_audio_duration - new_audio_duration
        silence = np.zeros(int(silence_duration * new_audio_clip.fps))  
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_silence_file:
            sf.write(temp_silence_file.name, silence, new_audio_clip.fps)
            silence_clip = mp.AudioFileClip(temp_silence_file.name)
        
        new_audio_clip = mp.concatenate_audioclips([new_audio_clip, silence_clip])

    elif new_audio_duration > original_audio_duration:
        # If the audio is longer, trim the audio to match the video's audio duration
        new_audio_clip = new_audio_clip.subclip(0, original_audio_duration)

    video_clip = video_clip.set_audio(new_audio_clip)

    # Save the video with the new audio
    temp_output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output_video.close()

    video_clip.write_videofile(temp_output_video.name, codec="libx264", audio_codec="aac")
    
    return temp_output_video.name  

st.title("Video Audio Replacement with AI")
video_file = st.file_uploader("Upload your video file", type=["mp4", "mov"])

if video_file is not None:
    st.write("Video File Selected")

    # Step 1: Extract audio and transcribe
    if st.button("Transcribe and Correct"):
        st.success("Video File Uploaded Successfully")
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_file.write(video_file.read())
        temp_video_file.close()

        # Extract the audio from video using moviepy
        video_clip = mp.VideoFileClip(temp_video_file.name)
        audio_file_path = "extracted_audio.wav"
        video_clip.audio.write_audiofile(audio_file_path)

        transcript = transcribe_audio_with_timestamps(audio_file_path)
        if transcript is None:
            st.error("Transcription failed. Please try again.")
        else:
            transcription = transcript.text
            word_timestamps = transcript.words 

               
            corrected_transcription = correct_transcription(transcription)
            st.write("New Transcriptions:",corrected_transcription)

            aligned_timestamps = map_corrected_transcription_to_timestamps(transcription, corrected_transcription, word_timestamps)

            audio_path = synthesize_speech_with_timeline(corrected_transcription, aligned_timestamps)

            final_video_path = replace_audio_in_video_with_timeline(temp_video_file.name, audio_path, aligned_timestamps)

            col1, col2 = st.columns(2)  # Create two columns
            
            with col1:
                st.subheader("Original Video")
                st.video(temp_video_file.name)  # Show original video

            with col2:
                st.subheader("Processed Video")
                st.video(final_video_path)  # Show processed video

            st.success("Video successfully processed with corrected audio!")








