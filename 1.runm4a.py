import whisper
import os
audio_file = "test/"
filelsit = os.listdir(audio_file)
model = whisper.load_model("small")
msglist = []
for name in filelsit:
    if name[-4:] != ".m4a":
        continue
    filename = audio_file+name
    result = model.transcribe(filename,language='zh',initial_prompt='替代役，節能家電汰換補助')
    # result = model.transcribe(filename,language='zh')
    msglist.append(f'name:{name} msg:{result["text"]}')

for msg in msglist:
    print(msg)
