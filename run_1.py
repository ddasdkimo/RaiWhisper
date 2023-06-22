# 由 openai 訓練的英文語音轉文字模型測試
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
# # 英文
# model_name = 'openai/whisper-tiny.en'
# audio_file = "2021-09-01_001.wav"
# language = "en"

# 中文
model_name = 'openai/whisper-tiny'
audio_file = "chinese2_32.wav"
language = "zh"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = language, task = "transcribe")
speech, _ = librosa.load(audio_file)
input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
predicted_ids = model.generate(input_features, max_length=448)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)