import whisper

model = whisper.load_model("turbo")
result = model.transcribe("viclab/viclab/voice/sample/01000120.wav")
print(result["text"])