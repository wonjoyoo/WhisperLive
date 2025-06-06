import json
import logging
import threading
import time
import torch

from whisper_live.transcriber.transcriber_faster_whisper import WhisperModel
from whisper_live.backend.base import ServeClientBase


class ServeClientFasterWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="small.en",
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
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe". Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): The whisper model size. Defaults to 'small.en'
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.

        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
        )
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
            "distil-medium.en", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo"
        ]

        self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path.endswith("en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            self.compute_type = "float16" if major >= 7 else "float32"
        else:
            self.compute_type = "int8"

        if self.model_size_or_path is None:
            return
        logging.info(f"Using Device={device} with precision {self.compute_type}")
    
        try:
            if single_model:
                if ServeClientFasterWhisper.SINGLE_MODEL is None:
                    self.create_model(device)
                    ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
                else:
                    self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
            else:
                self.create_model(device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        self.use_vad = use_vad

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        if self.websocket:
            self.websocket.send(
                json.dumps(
                    {
                        "uid": self.client_uid,
                        "message": self.SERVER_READY,
                        "backend": "faster_whisper"
                    }
                )
            )

    def create_model(self, device):
        """
        Instantiates a new model, sets it as the transcriber.
        """
        self.transcriber = WhisperModel(
            self.model_size_or_path,
            device=device,
            compute_type=self.compute_type,
            local_files_only=False,
        )

    def check_valid_model(self, model_size):
        """
        Check if it's a valid whisper model size.

        Args:
            model_size (str): The name of the model size to check.

        Returns:
            str: The model size if valid, None otherwise.
        """
        if model_size not in self.model_sizes:
            self.websocket.send(
                json.dumps(
                    {
                        "uid": self.client_uid,
                        "status": "ERROR",
                        "message": f"Invalid model size {model_size}. Available choices: {self.model_sizes}"
                    }
                )
            )
            return None
        return model_size

    def set_language(self, info):
        """
        Updates the language attribute based on the detected language information.

        Args:
            info (object): An object containing the detected language and its probability. This object
                        must have at least two attributes: `language`, a string indicating the detected
                        language, and `language_probability`, a float representing the confidence level
                        of the language detection.
        """
        if info.language_probability > 0.5:
            self.language = info.language
            logging.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        result, info = self.transcriber.transcribe(
            input_sample,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None,
            beam_size=1,  # 🔽 낮추면 빠름, 하지만 품질 저하 가능
            condition_on_previous_text=False  # 🔽 독립적 문장 추론
            )
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        #print("handle_transcription_output: segments초기화 ")
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            self._last_whisper_segment = last_segment
            #print('s=' , segments)
            segments = self.prepare_segments(last_segment)
            #print('s=' , segments)
            #print('l=' , last_segment)
            #print(last_segment)
            #print(result)


        if len(segments):
            self.send_transcription_to_client(segments)

        #self.prompt = ' '.join(segment['text'] for segment in segments)
        #self.transcription_queue.put({"uid": self.client_uid, "prompt": self.text, "eos": self.eos})

