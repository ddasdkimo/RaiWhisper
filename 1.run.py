# 由 openai 訓練的英文語音轉文字模型測試
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import time
import os

# # 英文
# model_name = 'openai/whisper-tiny.en'
# audio_file = "2021-09-01_001.wav"
# language = "en"

# 中文
# model_name = 'openai/whisper-tiny'
# model_name = 'openai/whisper-base'
# model_name = 'openai/whisper-small'
# model_name = 'openai/whisper-medium'
model_name = 'openai/whisper-large'
audio_file = "test/"
filelsit = os.listdir(audio_file)
language = "zh"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = language, task = "transcribe")
for name in filelsit:
    if name[-4:] != ".m4a":
        continue
    speech, _ = librosa.load(audio_file+name)
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
    t1 = time.time()
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
    predicted_ids = model.generate(input_features, max_length=448)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    t2 = time.time()
    print(f'filename:{name} msg:{transcription}')
    