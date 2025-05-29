import json
import logging
import threading
import numpy as np

from whisper_live.backend.base import ServeClientBase
from whisper_live.transcriber.transcriber_rknn import RKNNWhisperModel


class ServeClientRknn(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model=None,
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        transcription_queue=None,
    ):
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            transcription_queue=transcription_queue,
        )
        self.language = language or "en"
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}
        self.use_vad = use_vad

        self.model_paths = {
            "encoder": "~/rknn_model_zoo/examples/whisper/model/whisper_encoder_base_20s.rknn",
            "decoder": "~/rknn_model_zoo/examples/whisper/model/whisper_decoder_base_20s.rknn",
            "vocab": "~/rknn_model_zoo/examples/whisper/model/vocab_en.txt"
        }

        try:
            if single_model:
                if ServeClientRknn.SINGLE_MODEL is None:
                    self.create_model()
                    ServeClientRknn.SINGLE_MODEL = self.transcriber
                else:
                    self.transcriber = ServeClientRknn.SINGLE_MODEL
            else:
                self.create_model()
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model"
            }))
            self.websocket.close()
            return

        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        if self.websocket:
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "message": self.SERVER_READY,
                "backend": "rknn"
            }))

    def create_model(self):
        self.transcriber = RKNNWhisperModel(
            encoder_path=self.model_paths["encoder"],
            decoder_path=self.model_paths["decoder"],
            vocab_path=self.model_paths["vocab"]
        )

    def transcribe_audio(self, input_sample: np.ndarray):
        if ServeClientRknn.SINGLE_MODEL:
            ServeClientRknn.SINGLE_MODEL_LOCK.acquire()
        result, info = self.transcriber.transcribe(
            input_sample,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters
        )
        if ServeClientRknn.SINGLE_MODEL:
            ServeClientRknn.SINGLE_MODEL_LOCK.release()

        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def handle_transcription_output(self, result, duration):
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            self._last_whisper_segment = last_segment
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
