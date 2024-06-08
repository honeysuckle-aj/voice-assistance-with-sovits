import argparse
import time
import wave
import queue
import struct
import threading
import subprocess
import torch
import warnings
import keyboard
import pyaudio
import simpleaudio as sa
import whisper

from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager

from sovits_tools.voice import get_tts_wav, load_sovits_weights, load_gpt_weights

# import voice configs for tts
from voice_configs import *  

warnings.filterwarnings("ignore", category=UserWarning)

LANG = "CN"  # CN for Chinese, EN for English
DEBUG = True

# Model Configuration
WHISP_PATH = "models/whisper-large-v3"
# MODEL_PATH = "models/yi-chat-6b.Q8_0.gguf"  # Or models/yi-chat-6b.Q8_0.gguf

# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * RATE / CHUNK  # two seconds of silence marks the end of user voice input
MIC_IDX = 1  # Set microphone id. Use tools/list_microphones.py to see a device list.
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms


def get_language(text):
    zh = False
    en = False
    for w in text:
        if 'a' <= w <= 'z' or 'A' <= w <= 'Z':
            en = True
            break
    for w in text:
        if '\u4e00' <= w <= '\u9fff':
            zh = True
            break
    if zh and not en:
        return "all_zh"
    if zh and en:
        return "zh"
    return "en"


def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX,
                        frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []
    keyboard.wait("space")
    print("Listening...")
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open('recordings/output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def main(args):
    prompt_path = f"prompts/{args.config}_template.txt"
    with open(prompt_path, 'r', encoding='utf-8') as file:
        template = file.read().strip()  # {dialogue}
    prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])
    vcs = eval(args.config + "_configs")
    sovits_path = vcs.SOVITS_PATH
    gpt_path = vcs.GPT_PATH
    ref_wav_path = vcs.REF_WAVE_PATH
    with open(vcs.REF_TEXT_PATH, "r", encoding='utf-8') as f:
        ref_text = f.read()

    # Create an instance of the VoiceOutputCallbackHandler
    # voice_output_handler = VoiceOutputCallbackHandler()

    # Create a callback manager with the voice output handler
    # callback_manager = BaseCallbackManager(handlers=[voice_output_handler])

    # llm = LlamaCpp(
    #     model_path=MODEL_PATH,
    #     n_gpu_layers=6,  # Metal set to 1 is enough.
    #     n_batch=512,  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    #     n_ctx=4096,  # Update the context window size to 4096
    #     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    #     callback_manager=callback_manager,
    #     stop=["<|im_end|>"],
    #     verbose=False,
    # )
    llm = Ollama(model="qwen:7b", verbose=False)
    whisper_model = whisper.load_model("large", download_root="./models")
    vq_model, hps = load_sovits_weights(sovits_path)
    tts_model = load_gpt_weights(gpt_path)
    # llm.to(device)
    dialogue = ""
    try:
        while True:
            # if voice_output_handler.tts_busy:  # Check if TTS is busy
            #     continue  # Skip to the next iteration if TTS is busy
            try:
                print("按下空格开始说话")
                # user_input = input()
                record_audio()
                print("Transcribing...")
                time_ckpt = time.time()
                user_input = whisper_model.transcribe("recordings/output.wav", language="zh")["text"]
                
                print("%s: %s (Time %d ms)" % ("Guest", user_input, (time.time() - time_ckpt) * 1000))

            except subprocess.CalledProcessError:
                print("voice recognition failed, please try again")
                continue
            time_ckpt = time.time()
            print("Generating...")
            dialogue += "*Q* {}\n".format(user_input)
            prompt = prompt_template.format(dialogue=dialogue)
            reply = llm.invoke(prompt, max_tokens=4096)
            if reply is not None:
                # voice_output_handler.speech_queue.put(reply)
                text_lang = get_language(reply)

                sampling_rate, output_wave = get_tts_wav(vq_model, tts_model, hps, ref_wav_path, ref_text, reply,
                                                         text_language=text_lang)
                audio_obj = sa.WaveObject(output_wave, sample_rate=sampling_rate // 2)
                play_obj = audio_obj.play()
                play_obj.wait_done()
                dialogue += "*A* {}\n".format(reply)
                print("%s: %s (Time %d ms)" % ("Server", reply.strip(), (time.time() - time_ckpt) * 1000))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    # if LANG == "CN":
    #     prompt_path = "prompts/example-cn.txt"
    # else:
    #     prompt_path = "prompts/example-en.txt"
    # with open(prompt_path, 'r', encoding='utf-8') as file:
    #     template = file.read().strip()  # {dialogue}
    # prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"cfg0")
    args = parser.parse_args()
    main(args)

