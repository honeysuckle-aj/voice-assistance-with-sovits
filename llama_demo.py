import io
import time
import wave

import requests
import simpleaudio as sa
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.ollama import Ollama

llm = Ollama(model="qwen:7b", )


def get_completion_ollama(prompt):
    return llm.invoke(prompt)

def generate_speech(text, port):
    time_ckpt = time.time()
    data={
        "text": text,
        "text_language": "zh"
    }
    response = requests.post("http://127.0.0.1:{}".format(port), json=data)
    if response.status_code == 400:
        raise Exception(f"GPT-SoVITS ERROR: {response.message}")
    audio_data = io.BytesIO(response.content)
    with wave.open(audio_data, 'rb') as wave_read:
        audio_frames = wave_read.readframes(wave_read.getnframes())
        audio_wave_obj = sa.WaveObject(audio_frames, wave_read.getnchannels(), wave_read.getsampwidth(), wave_read.getframerate())
    play_obj = audio_wave_obj.play()
    play_obj.wait_done()
    print("Audio Generation Time: %d ms\n" % ((time.time() - time_ckpt) * 1000))

prompt = '你是谁'
reply = llm(prompt, max_tokens=4096)
print(reply)
generate_speech(reply, port=9880)
# print(res)