import speech_recognition as sr
from os import path
from pydub import AudioSegment
from transformers import pipeline
import os
import moviepy.editor as mp
 
# Insert Local Video File Path
clip = mp.VideoFileClip(r"D:\demo.mp4")
 
# Insert Local Audio File Path
clip.audio.write_audiofile(r"demo_audio.wav")


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

AUDIO_FILE = "demo_audio.wav"

# # use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file
        text_output=r.recognize_google(audio)

length = len(text_output.split(" ")) + 1
max_length=(length)//4
min_length=(length)//5
print(max_length, min_length)
print(summarizer(text_output, max_length=max_length,min_length=min_length, do_sample=False))