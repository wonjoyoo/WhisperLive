import threading
import os
import signal
import sys
import time
import queue

import numpy as np
import sounddevice as sd

from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
#from llm_service import TensorRTLLMEngine
#from tts_service import WhisperSpeechTTS

# ── 1) GLOBAL SHUTDOWN HANDLER ───────────────────────────────────────
shutdown_flag = False

def _on_sigint(signum, frame):
    global shutdown_flag
    print("\n🛑 SIGINT received, shutting down…")
    shutdown_flag = True

signal.signal(signal.SIGINT, _on_sigint)

# ── 2) QUEUE 기반 파이프라인 준비 ────────────────────────────────────
transcription_queue = queue.Queue()
llm_queue = queue.Queue()
audio_queue = queue.Queue()

# ── 3) LOCAL WHISPER CLIENT 설정 ────────────────────────────────────
class LocalServeClientFasterWhisper(ServeClientFasterWhisper):
    def __init__(self, *args, transcription_queue=None, **kwargs):
        # websocket 비활성화, client_uid 고정
        kwargs["websocket"] = None
        kwargs["client_uid"] = "localtest"
        super().__init__(*args, **kwargs)
        self.transcription_queue = transcription_queue

    def websocket_send(self, *args, **kwargs):
        pass

    def websocket_close(self):
        pass

    def send_transcription_to_client(self, segments):  ## override
        print(segments)
        # segments: 리스트 of {'text': ...}
        prompt = " ".join([seg['text'] for seg in segments])
        
        if self.last_prompt != prompt:
            #print(prompt)
            print(self._last_whisper_segment['text'])
            self.last_prompt = prompt
            # eos 플래그는 self.eos에 담겨 있음
            self.transcription_queue.put({
                'prompt': prompt,
            })

# 클라이언트 생성
client = LocalServeClientFasterWhisper(
    model="small",
    language="ko",
    use_vad=True,
    single_model=True,
    vad_parameters={"onset": 0.3},
    transcription_queue=transcription_queue
)

'''
# ── 4) LLM 쓰레드 실행 ─────────────────────────────────────────────
def llm_thread_fn():
    # TensorRT LLM 엔진 초기화
    llm_engine = TensorRTLLMEngine()
    # 실제 run() 시그니처: (model_path, tokenizer_path, phi_model_type,
    #                       transcription_queue, llm_queue, audio_queue)
    llm_engine.run(
        client.model,               # 모델 경로 또는 이름
        client.language,            # 토크나이저 경로 또는 이름
        None,                       # phi_model_type (없으면 None)
        transcription_queue,
        llm_queue,
        audio_queue
    )

threading.Thread(target=llm_thread_fn, daemon=True).start()

# ── 5) TTS 쓰레드 실행 ─────────────────────────────────────────────
threading.Thread(
    target=WhisperSpeechTTS().run,
    args=("0.0.0.0", 8888, audio_queue),
    daemon=True
).start()
'''

# ── 6) 마이크 스트림 시작 ───────────────────────────────────────────
stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype="float32",
    callback=lambda indata, frames, time_info, status: client.add_frames(indata[:, 0].copy())
)
stream.start()

print("🎤 말하세요. (Ctrl+C로 종료)")
try:
    while not shutdown_flag:
        time.sleep(0.1)
    print("🛑 Shutdown flag set, cleaning up…")
finally:
    # 오디오 스트림 정리
    try:
        sd.stop()
        stream.close()
    except Exception:
        pass

    # Whisper 클라이언트 종료
    try:
        client.set_eos(True)
        client.disconnect()
        client.cleanup()
    except Exception:
        pass

    print("✅ All resources cleaned up, exiting now.")
    os._exit(0)  # 모든 쓰레드 강제 종료
