import os
# import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# make a new models directory
MODEL_PATH = './models/'
os.mkdir(MODEL_PATH)

# load Wav2Vec2 tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

tokenizer.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)