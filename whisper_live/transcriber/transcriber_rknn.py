from rknn.api import RKNN
import numpy as np
import torch
import torch.nn.functional as F
import scipy
from whisper_live.utils import Segment  # 추가


def read_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = {}
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 1:
                vocab[parts[0]] = ""
            else:
                vocab[parts[0]] = parts[1]
        return vocab

def init_model(model_path):
    model = RKNN()
    if model.load_rknn(model_path) != 0:
        raise RuntimeError(f"Failed to load RKNN model: {model_path}")

    # target을 명시적으로 설정 (Orange Pi는 일반적으로 'rk3588')
    if model.init_runtime(target='rk3588') != 0:
        raise RuntimeError(f"Failed to init RKNN runtime for: {model_path}")
    return model


def run_encoder(encoder_model, x):
    return encoder_model.inference(inputs=[x])[0]  # ✅ 리스트로 감싸기

def run_decoder(decoder_model, tokens, out_encoder):
    return decoder_model.inference(inputs=[np.asarray([tokens], dtype="int64"), out_encoder])[0]  # ✅ 명시적 리스트


def log_mel_spectrogram(audio, n_mels=80):
    N_FFT = 400
    HOP_LENGTH = 160
    window = torch.hann_window(N_FFT)
    stft = torch.stft(torch.from_numpy(audio), N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = np.loadtxt("/home/orangepi/rknn_model_zoo/examples/whisper/model/mel_80_filters.txt", dtype=np.float32).reshape((80, 201))
    mel_spec = torch.from_numpy(filters) @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def pad_or_trim(mel):
    MAX_LENGTH = 2000
    padded = np.zeros((mel.shape[0], MAX_LENGTH), dtype=np.float32)
    length = min(mel.shape[1], MAX_LENGTH)
    padded[:, :length] = mel[:, :length]
    return padded


class RKNNWhisperModel:
    def __init__(self, encoder_path, decoder_path, vocab_path):
        self.encoder_model = init_model(encoder_path)
        self.decoder_model = init_model(decoder_path)
        self.vocab = read_vocab(vocab_path)
        self.task_code = 50259  # English task code

    def transcribe(self, audio_np, **kwargs):
        mel = log_mel_spectrogram(audio_np, n_mels=80).numpy()
        x_mel = pad_or_trim(mel)[None, ...]
        encoded = run_encoder(self.encoder_model, x_mel)
        tokens = [50258, self.task_code, 50359, 50363] * 3

        result_str = ""
        end_token = 50257
        timestamp_begin = 50364
        max_tokens = 12
        pop_id = max_tokens

        while True:
            out = run_decoder(self.decoder_model, tokens, encoded)
            next_token = out[0, -1].argmax()
            if next_token == end_token:
                break
            if next_token > timestamp_begin:
                continue
            if pop_id > 4:
                pop_id -= 1
            tokens.pop(pop_id)
            tokens.append(next_token)
            result_str += self.vocab.get(str(next_token), '')

        segment = Segment(
            id=0,
            seek=0,
            start=0.0,
            end=2.0,
            text=result_str,
            tokens=tokens,
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0
        )

        info = {
            "language": "en",
            "language_probability": 1.0
        }

        return [segment], info
