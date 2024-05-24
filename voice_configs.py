import os
MODEL_ROOT = r"D:\Projects\voice-assistant\models"
REF_ROOT = r"D:\Projects\voice-assistant\models\Paimon\reference"
GPT_PATH = os.path.join(MODEL_ROOT, "paimeng-gpt.ckpt")
SOVITS_PATH = os.path.join(MODEL_ROOT, "paimeng-sovits.pth")
CUHUBERT_PATH = os.path.join(MODEL_ROOT, "chinese-hubert-base")
BERT_PATH = os.path.join(MODEL_ROOT, "chinese-roberta-wwm-ext-large")
REF_WAVE_PATH = os.path.join(REF_ROOT, "shuohua_jiran.wav")
REF_TEXT = "既然罗莎莉亚说足迹上有元素力，用元素视野应该能很清楚地看到吧"

