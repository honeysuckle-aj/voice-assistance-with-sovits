import os

MODEL_ROOT = "models"
SPEAKER = "paimon"
GPT_PATH = os.path.join(MODEL_ROOT, SPEAKER, "gpt.ckpt")
SOVITS_PATH = os.path.join(MODEL_ROOT, SPEAKER, "sovits.pth")
CUHUBERT_PATH = os.path.join(MODEL_ROOT, "chinese-hubert-base")
BERT_PATH = os.path.join(MODEL_ROOT, "chinese-roberta-wwm-ext-large")
REF_WAVE_PATH = os.path.join(MODEL_ROOT, SPEAKER, "ref_wave.wav")
REF_TEXT_PATH = os.path.join(MODEL_ROOT, SPEAKER, "ref.txt")
