from whisper_live import WhisperSTT

stt = WhisperSTT(model="base")
for text in stt.stream():
    print(f"Recognized: {text}")
