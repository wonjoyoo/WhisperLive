from whisper_live.client import TranscriptionClient

# WhisperLive 서버에 연결
client = TranscriptionClient(
    host="localhost",              # 또는 서버 IP
    port=9090,                     # 서버 포트
    lang="ko",                     # 한국어 음성의 경우 "ko"
    translate=False,              # 번역 필요 없으면 False (원어 그대로)
    model="small",                # small, medium 등 원하는 faster-whisper 모델 이름
    use_vad=True,                 # VAD 사용 여부 (음성만 감지)
    save_output_recording=False,   # 입력을 .wav 파일로 저장할지 여부
    output_recording_filename="mic_input.wav",  # 저장될 파일 이름
    max_connection_time=100000,
)

# 마이크에서 음성 입력 -> 실시간 전송 -> 출력
client()
