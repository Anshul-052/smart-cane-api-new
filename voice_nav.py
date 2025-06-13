import speech_recognition as sr
import googlemaps
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
import re
import time

def get_destination_by_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🔊 Speak your destination...")
        audio = r.listen(source)

    try:
        destination = r.recognize_google(audio)
        print(f"📍 You said: {destination}")
        return destination
    except sr.UnknownValueError:
        print("⚠️ Sorry, I couldn't understand.")
        return None
    except sr.RequestError as e:
        print(f"⚠️ API Error: {e}")
        return None

def get_directions(origin, destination, api_key):
    gmaps = googlemaps.Client(key=api_key)
    directions_result = gmaps.directions(origin, destination, mode="walking")
    steps = directions_result[0]['legs'][0]['steps']
    directions_text = []

    for step in steps:
        instruction = re.sub('<[^<]+?>', '', step['html_instructions'])  # Remove HTML
        directions_text.append(instruction)

    return directions_text

def speak_instruction(text):
    tts = gTTS(text=text, lang='en')
    filename = "temp_instruction.mp3"
    tts.save(filename)

    audio = AudioSegment.from_mp3(filename)
    play(audio)

    os.remove(filename)

def wait_for_next_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Say 'next' for the next instruction...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        return command.lower() == "next"
    except sr.UnknownValueError:
        return False
    except sr.RequestError:
        return False

# --- Main Program ---

origin = "Vishwakarma Institute of Technology, Kondhwa Campus"
api_key = "AIzaSyAF37kDyerCWtnvCFxz8AqVsv_U_rRmBLI"

destination = get_destination_by_voice()

if destination:
    directions = get_directions(origin, destination, api_key)

    print("\n📢 Route directions will be given step-by-step...\n")

    index = 0
    while index < len(directions):
        print(f"🧭 Step {index + 1}: {directions[index]}")
        speak_instruction(directions[index])

        index += 1
        if index < len(directions):
            while not wait_for_next_command():
                print("❗ Please say 'next' to continue...")
                
