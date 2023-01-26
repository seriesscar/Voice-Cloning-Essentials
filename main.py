import sys

import ipywidgets as widgets
import numpy as np
import torch

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
from scipy.io.wavfile import write


encoder.load_model(Path("saved_models/default/encoder.pt"))
synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
vocoder.load_model(Path("saved_models/default/vocoder.pt"))

embedding = None
SAMPLE_RATE = 22050

def _compute_embedding(audio):
  global embedding
  embedding = None
  embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, SAMPLE_RATE))

_compute_embedding("samples/elon1.wav")

text = "A loop in Python is a way for the computer to repeat a set of instructions multiple times. It allows you to execute the same code over and over again each time with a different value "

#embedding = torch.from_numpy(embedding)

def synthesize(embed, text):
  print("Synthesizing new audio...")
  #with io.capture_output() as captured:
  specs = synthesizer.synthesize_spectrograms([text], [embedding])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
  # Saving the audio to a wav file
  write("generated_audio.wav", synthesizer.sample_rate, generated_wav.astype(np.float32))
  print("Audio saved as generated_audio.wav")
#  display(Audio(generated_wav, rate=synthesizer.sample_rate, autoplay=True))

synthesize(embedding, text)
