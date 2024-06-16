#!/usr/bin/env python3

import argparse
import queue
import sys
import sounddevice as sd
import wave
import threading
import time
import numpy as np

from vosk import Model, KaldiRecognizer

q = queue.Queue()
last_activity_time = time.time()
ENERGY_THRESHOLD = 300  # Adjusted to detect only human voice

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, audio_time, status):
    """This is called (from a separate thread) for each audio block."""
    global last_activity_time
    if status:
        print(status, file=sys.stderr)
    
    # Convert the input data to a numpy array of int16
    data_array = np.frombuffer(indata, dtype=np.int16)
    
    # Calculate average energy of the audio block (absolute mean amplitude)
    energy = np.mean(np.abs(data_array))
    
    if energy > ENERGY_THRESHOLD:
        last_activity_time = time.time()
        
    q.put(indata)

def record_audio(filename, samplerate, device, max_duration=7):
    global last_activity_time
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                           dtype="int16", channels=1, callback=callback):


        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)

            start_time = time.time()
            while time.time() - start_time < max_duration:
                data = q.get()
                wf.writeframes(data)
                current_time = time.time()

                # Check if no significant audio detected for 5 seconds
                if current_time - last_activity_time > 3:
                    print("No significant audio detected for 3 seconds. Stopping recording.")
                    break

  

def transcribe_audio(filename, model):
    wf = wave.open(filename, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    # Print final recognition result
    print(rec.FinalResult())

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l", "--list-devices", action="store_true",
    help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    "-f", "--filename", type=str, metavar="FILENAME", default="recording.wav",
    help="audio file to store recording to")
parser.add_argument(
    "-d", "--device", type=int_or_str,
    help="input device (numeric ID or substring)")
parser.add_argument(
    "-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument(
    "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
args = parser.parse_args(remaining)

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        args.samplerate = int(device_info["default_samplerate"])
        
    if args.model is None:
        model = Model(lang="en-us")
    else:
        model = Model(lang=args.model)

    record_audio(args.filename, args.samplerate, args.device)
    transcribe_audio(args.filename, model)

except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ": " + str(e))




