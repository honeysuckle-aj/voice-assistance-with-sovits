# A Simple Voice Assistant With GPT-SOVITS

English | [简体中文](README-CN.md)

This is a local llm assistant which use gpt-sovits to generate someone-like vocie.

This work is mainly based on LinYi's [voice-assistant](https://github.com/linyiLYi/voice-assistant) and RVC-Boss's [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).

### File Structure

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

This project is a single-script project, with main.py containing all program logic. The `voice_config.json` controls the configuration of generated voice. The `models/`folder stores model files, including llm, whisper and tts. `prompts/`contains prompt words. `recordings/`holds temporary recordings. `sovits_tools/list_microphones.py`is a simple script to view the microphone list, used in`main.py`to specify the microphone number. `sovits_tools/` contains voice generating tools.

## Installation Guide

It is a python project.

Before you install python requirements, you need to download and install [Ollama](https://ollama.com/download).

It is recommanded to use Anaconda to settle the environment.

### Environment Configuration

Create your conda env

```powershell
conda create -n chat python=3.10 -y
conda activate chat
```

Before install other packages, you need to install [Pytorch](https://pytorch.org/get-started/locally/) whose version depends on your device especially when you use GPU.

Then

```powershell
pip install -r requirements.txt
```

You can pull ollama models in advance

```powershell
ollama pull Qwen:7b
ollama run Qwen:7b
```

### Model Files

The model files are stored in the `models/` folder and specified in the script via the `MODEL_PATH` variable.

I use Qwen:7b and there are plenty of other llm models you can choose from [Ollama Library](https://ollama.com/library).

Thank you to all selfless programmers for their contributions to the open-source community!
