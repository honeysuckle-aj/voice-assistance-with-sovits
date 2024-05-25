# 极简语音助手脚本

简体中文 | [English](README.md)

这是一个简单的 Python 脚本项目，可以通过语音与本地大语言模型进行对话。

这个脚本主要依据林哥的[voice-assistant](https://github.com/linyiLYi/voice-assistant)和 RVC-Boss的[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

### 文件结构

```powershell
│  .gitignore
│  LICENSE
│  llama_demo.py
│  main.py
│  README-CN.md
│  README.md
│  requirements.txt
│  utils.py
│  voice_configs.py
│
├─logs
│  └─s2
│      └─big2k1
│              config.json
│
├─models
│  ├─chinese-hubert-base
│  │      config.json
│  │      preprocessor_config.json
│  │      pytorch_model.bin
│  │
│  ├─chinese-roberta-wwm-ext-large
│         config.json
│         pytorch_model.bin
│         tokenizer.json
│
├─prompts
│      example-cn.txt
│      example-en.txt
│      wpq_template.txt
│
├─recordings
│      output.wav
│
└─sovits_tools
    │  gweight.txt
    │  list_microphones.py
    │  my_utils.py
    │  utils.py
    │  voice.py
    │
    ├─AR
    │  │  __init__.py
    │  │
    │  ├─data
    │  │      bucket_sampler.py
    │  │      dataset.py
    │  │      data_module.py
    │  │      __init__.py
    │  │
    │  ├─models
    │  │     t2s_lightning_module.py
    │  │     t2s_lightning_module_onnx.py
    │  │     t2s_model.py
    │  │     t2s_model_onnx.py
    │  │     utils.py
    │  │     __init__.py
    │  │
    │  ├─modules
    │  │     activation.py
    │  │     activation_onnx.py
    │  │     embedding.py
    │  │     embedding_onnx.py
    │  │     lr_schedulers.py
    │  │     optim.py
    │  │     patched_mha_with_cache.py
    │  │     patched_mha_with_cache_onnx.py
    │  │     scaling.py
    │  │     transformer.py
    │  │     transformer_onnx.py
    │  │     __init__.py
    │  │
    │  ├─text_processing
    │  │      phonemizer.py
    │  │      symbols.py
    │  │      __init__.py
    │  │
    │  └─utils
    │          initialize.py
    │          io.py
    │          __init__.py
    │
    ├─configs
    │      s1.yaml
    │      s1big.yaml
    │      s1big2.yaml
    │      s1longer.yaml
    │      s1mq.yaml
    │      s2.json
    │      train.yaml
    │
    ├─feature_extractor
    │      cnhubert.py
    │      whisper_enc.py
    │      __init__.py
    │
    ├─module
    │      attentions.py
    │      attentions_onnx.py
    │      commons.py
    │      core_vq.py
    │      data_utils.py
    │      losses.py
    │      mel_processing.py
    │      models.py
    │      models_onnx.py
    │      modules.py
    │      mrte_model.py
    │      quantize.py
    │      transforms.py
    │      __init__.py
    │
    └─text
        │  chinese.py
        │  cleaner.py
        │  cmudict-fast.rep
        │  cmudict.rep
        │  engdict-hot.rep
        │  engdict_cache.pickle
        │  english.py
        │  japanese.py
        │  opencpop-strict.txt
        │  symbols.py
        │  tone_sandhi.py
        │  __init__.py
        │
        └─zh_normalization
                char_convert.py
                chronology.py
                constants.py
                num.py
                phonecode.py
                quantifier.py
                README.md
                text_normlization.py
                __init__.py
```

本项目为单脚本项目，主要程序逻辑全部在 `main.py` 中。`voice_config.json`控制所需的语音模型。 `models/` 文件夹存放模型文件。`prompts/` 存放提示词。`recordings/` 存放临时录音。`sovits_tools/list_microphones.py`是一个用来查看麦克风列表的简单脚本，用来在`main.py` 中指定麦克风序号。`sovits_tools` 中为语音生成的代码。

## 安装指南

这是一个python工程。

在安装python依赖之前，你需要先下载安装[Ollama](https://ollama.com/download)。

推荐使用Anaconda安装依赖包。

## 环境配置

创建环境

```powershell
conda create -n chat python=3.10 -y
conda activate chat
```

在安装其他依赖之前，请根据自己的显卡设备安装[Pytorch](https://pytorch.org/get-started/locally/)。

然后

```powershell
pip install -r requirements.txt
```

你可以先下载一些大预言模型到本地

```powershell
ollama pull Qwen:7b
ollama run Qwen:7b
```

### 模型文件

本项目的语音识别部分基于 OpenAI 的 whisper 模型，

大预言模型基于通义千问，你可以在[Ollama Library](https://ollama.com/library)中选择其他模型。

感谢各位程序工作者对开源社区的贡献！
