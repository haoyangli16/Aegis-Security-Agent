import queue
import threading
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# 配置模型
model = WhisperModel("small", compute_type="int8")  # 支持 tiny, base, small, medium, large

# 音频设置
RATE = 16000
CHUNK = 1024  # 每次读取的音频帧数
CHANNELS = 1
FORMAT = pyaudio.paInt16

audio_queue = queue.Queue()

# 麦克风录音线程
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("🎤 开始监听麦克风... 按 Ctrl+C 退出")
    try:
        while True:
            data = stream.read(CHUNK)
            audio_queue.put(data)
    except KeyboardInterrupt:
        print("🛑 停止录音")
        stream.stop_stream()
        stream.close()
        p.terminate()

# 推理线程
def transcribe_streaming():
    buffer = b""
    sample_limit = RATE * 5  # 每 5 秒跑一次推理
    frame_count = 0

    while True:
        data = audio_queue.get()
        buffer += data
        frame_count += len(data)

        if frame_count >= sample_limit * 2:  # int16 每个样本2字节
            # 转换为 float32
            audio_np = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0

            segments, _ = model.transcribe(audio_np, language="zh", beam_size=5)
            for seg in segments:
                print(f"[{seg.start:.2f}s - {seg.end:.2f}s]: {seg.text}")

            buffer = b""
            frame_count = 0

# 启动两个线程
recording_thread = threading.Thread(target=record_audio)
transcription_thread = threading.Thread(target=transcribe_streaming)

recording_thread.start()
transcription_thread.start()
