import streamlit as st
import os
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

DATA_PATH = './data/'
pre_loaded_clips = tuple(os.listdir(DATA_PATH))
transcription = '/// No Transcription ///'

################################################################################
@st.cache
def load_tokenizer():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer
    
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model():
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return model

def transcribe(model, tokenizer, audio_clip_path, sample_rate:int = 16000):
    audio, rate = librosa.load(audio_clip_path, sr = sample_rate)
    input_values = tokenizer(audio, return_tensors = "pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    transcription = tokenizer.batch_decode(prediction)[0]
    return transcription

################################################################################

st.title('Skope Speech-to-Text Demo')

################################################################################

with st.spinner("Loading NLP model..."):
    tokenizer = load_tokenizer()
    model = load_model()

################################################################################

# Choose preloaded or uploaded sound clip
preloaded = st.sidebar.radio('Preloaded', ("Yes", "No"))

################################################################################

# Choose a sound clip already saved to disk
if preloaded == 'Yes':
    clip = st.sidebar.selectbox('Choose an example audio clip', pre_loaded_clips)
    PATH_TO_CLIP = os.path.join(DATA_PATH, clip)    

load_audio = st.sidebar.button('Load Audio')

################################################################################

# audio player
if not load_audio: 
    st.write('Select audio clip from sidebar, and click \'Load Audio\'')
else:
    audio_file = open(PATH_TO_CLIP, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    with st.spinner('Making transcript...'):
        transcription = transcribe(model, tokenizer, PATH_TO_CLIP, 16000) # hard code sample rate    
        st.write('\n' + transcription)


################################################################################




        
    
################################################################################
