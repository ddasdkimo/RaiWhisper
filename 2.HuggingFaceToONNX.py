from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import librosa

audio_file = "chinese2_32.wav"
model_name = 'openai/whisper-tiny'
language = "zh"

processor = WhisperProcessor.from_pretrained(model_name)
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=True)
speech, _ = librosa.load(audio_file)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = language, task = "transcribe")
input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
predicted_ids = model.generate(input_features, max_length=448)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)