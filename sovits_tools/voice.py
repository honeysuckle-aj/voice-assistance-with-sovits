import os, re
import LangSegment
from tqdm import tqdm

import voice_configs.wpq_configs as configs
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
import simpleaudio as sa
from time import time as ttime

from sovits_tools.feature_extractor import cnhubert
from sovits_tools.module.models import SynthesizerTrn
from sovits_tools.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from sovits_tools.text import cleaned_text_to_sequence
from sovits_tools.text.cleaner import clean_text
from sovits_tools.module.mel_processing import spectrogram_torch
from sovits_tools.my_utils import load_audio

gpt_path = configs.GPT_PATH
sovits_path = configs.SOVITS_PATH
cnhubert.cnhubert_base_path = configs.CUHUBERT_PATH
# cnhubert_base_path = configs.CUHUBERT_PATH
bert_path = configs.BERT_PATH

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
bert_model = bert_model.half().to(device)
ssl_model = cnhubert.get_model().half().to(device)

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def load_sovits_weights(sovits_path):
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    # if ("pretrained" not in sovits_path):
    #     del vq_model.enc_q
    vq_model = vq_model.half().to(device).eval()

    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    return vq_model, hps





def load_gpt_weights(gpt_path):
    global max_sec, config
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    # with open("./gweight.txt", "w", encoding="utf-8") as f:
    #     f.write(gpt_path)
    return t2s_model





def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist) - 1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i - 1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i + 1]:
            textlist[i] += textlist[i + 1]
            del textlist[i + 1]
            del langlist[i + 1]
        else:
            i += 1

    return textlist, langlist


def clean_text_inf(text, language):
    formattext = ""
    language = language.replace("all_", "")
    for tmp in LangSegment.getTexts(text):
        if language == "ja":
            if tmp["lang"] == language or tmp["lang"] == "zh":
                formattext += tmp["text"] + " "
            continue
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16
        ).to(device)

    return bert


def nonen_clean_text_inf(text, language):
    if (language != "auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist = []
        langlist = []
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    # print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    if (language != "auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist = []
        langlist = []
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    # print(textlist)
    # print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert





def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_cleaned_text_final(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja", "auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    else:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text


def get_bert_final(phones, word2ph, text, language, device):
    if language == "en":
        bert = get_bert_inf(phones, word2ph, text, language)
    elif language in {"zh", "ja", "auto"}:
        bert = nonen_get_bert_inf(text, language)
    elif language == "all_zh":
        bert = get_bert_feature(text, word2ph).to(device)
    else:
        bert = torch.zeros((1024, len(phones))).to(device)
    return bert


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def get_tts_wav(vq_model, t2s_model, hps, ref_wav_path, prompt_text, text, top_k=20,
                top_p=0.6, temperature=0.6, hz=50, text_language="all_zh"):

    t0 = ttime()
    prompt_language = "all_zh"
    # text_language = "all_zh"

    # 输入的参考文本
    prompt_text = prompt_text.strip("\n")
    if prompt_text[-1] not in splits:
        prompt_text += "。" if prompt_language != "en" else "."

    text = text.strip("\n")

    if text[0] not in splits and len(get_first(text)) < 4:
        # 转换为目标文本
        text = "。" + text if text_language != "en" else "." + text

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16
    )
    with torch.no_grad():
        # 载入目标文本
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)

        wav16k = wav16k.half().to(device)
        zero_wav_torch = zero_wav_torch.half().to(device)

        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)

        prompt_semantic = codes[0, 0]
    t1 = ttime()
    # 按标点符号切分目标句子
    # text = cut_4_sentences(text)
    text = cut_by_punctuation(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []

    # 参考文本
    phones1, word2ph1, norm_text1 = get_cleaned_text_final(prompt_text, prompt_language)

    bert1 = get_bert_final(phones1, word2ph1, norm_text1, prompt_language, device).to(torch.float16)

    for text in tqdm(texts):
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        # print(i18n("实际输入的目标文本(每句):"), text)
        phones2, word2ph2, norm_text2 = get_cleaned_text_final(text, text_language)
        # print(i18n("前端处理后的文本(每句):"), norm_text2)
        bert2 = get_bert_final(phones2, word2ph2, norm_text2, text_language, device).to(torch.float16)

        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)


        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path).half().to(device)  # .to(device)

        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    # print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    return hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut_by_punctuation(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：]'
    # punds = r'[.;?!。？！;]'
    items = re.split(f'({punds})', inp)
    items = ["".join(group) for group in zip(items[::2], items[1::2])]
    opt = "\n".join(items)
    return opt

def cut_4_sentences(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

if __name__ == "__main__":
    MODEL_ROOT = r"D:\Projects\voice-assistant\models"
    REF_ROOT = r"D:\Projects\voice-assistant\models\Paimon\reference"
    GPT_PATH = os.path.join(MODEL_ROOT, "paimeng-gpt.ckpt")
    SOVITS_PATH = os.path.join(MODEL_ROOT, "paimeng-sovits.pth")
    CUHUBERT_PATH = os.path.join(MODEL_ROOT, "chinese-hubert-base")
    BERT_PATH = os.path.join(MODEL_ROOT, "chinese-roberta-wwm-ext-large")
    REF_WAVE_PATH = os.path.join(REF_ROOT, "shuohua_jiran.wav")
    REF_TEXT = "既然罗莎莉亚说足迹上有元素力，用元素视野应该能很清楚地看到吧"
    input_text = "你好，我叫派蒙，我会成为你最好的伙伴"
    vq_model, hps = load_sovits_weights(sovits_path)
    tts_model = load_gpt_weights(gpt_path)

    sampling_rate, output_wave = get_tts_wav(vq_model, tts_model, hps, REF_WAVE_PATH, REF_TEXT, input_text)
    audio_obj = sa.WaveObject(output_wave, sample_rate=sampling_rate//2)
    play_obj = audio_obj.play()
    play_obj.wait_done()
    print("done")
