import os
from whisper_live.client import TranscriptionClient


class AudioStreamHandler:
    """
    Handles real-time audio streams for transcription.
    This class is a wrapper around whisper_live.client.TranscriptionClient
    for handling real-time audio from a microphone, RTSP, or HLS stream.
    A WhisperLive server must be running to use this class.
    """

    def __init__(self, host="localhost", port=9090, model_size="small", lang="en", use_vad=True, **kwargs):
        """
        Initializes the AudioStreamHandler.
        Args:
            host (str): Hostname of the WhisperLive server.
            port (int): Port of the WhisperLive server.
            model_size (str): The size of the Whisper model to use (e.g., "tiny", "small", "medium", "large").
            lang (str): Language code for transcription (e.g., "en" for English).
            use_vad (bool): Whether to use Voice Activity Detection.
            **kwargs: Additional arguments for TranscriptionClient.
        """
        self.client = TranscriptionClient(
            host,
            port,
            model=model_size,
            lang=lang,
            use_vad=use_vad,
            **kwargs
        )

    def from_microphone(self):
        """
        Transcribes audio from the default microphone.
        This is a blocking call that will listen to the microphone until interrupted.
        """
        print("Listening to microphone...")
        self.client()

    def from_rtsp_stream(self, rtsp_url):
        """
        Transcribes an RTSP audio stream.
        Args:
            rtsp_url (str): The URL of the RTSP stream.
        """
        print(f"Listening to RTSP stream: {rtsp_url}")
        self.client(rtsp_url=rtsp_url)

    def from_hls_stream(self, hls_url):
        """
        Transcribes an HLS audio stream.
        Args:
            hls_url (str): The URL of the HLS stream.
        """
        print(f"Listening to HLS stream: {hls_url}")
        self.client(hls_url=hls_url)


class VoiceToText:
    """
    A class to perform voice-to-text transcription using a WhisperLive server.
    This class can transcribe local audio files and real-time audio streams.
    A WhisperLive server (with a backend like faster-whisper) must be running
    for this class to work.
    """

    def __init__(self, host="localhost", port=9090, model_size="small", lang="en", use_vad=True, **kwargs):
        """
        Initializes the VoiceToText client.
        Args:
            host (str): Hostname of the WhisperLive server.
            port (int): Port of the WhisperLive server.
            model_size (str): The size of the Whisper model to use (e.g., "tiny", "small", "medium", "large").
            lang (str): Language code for transcription (e.g., "en" for English).
            use_vad (bool): Whether to use Voice Activity Detection.
            **kwargs: Additional arguments for TranscriptionClient.
        """
        self.client = TranscriptionClient(
            host,
            port,
            model=model_size,
            lang=lang,
            use_vad=use_vad,
            **kwargs
        )
        self.stream_handler = AudioStreamHandler(host, port, model_size, lang, use_vad, **kwargs)

    def from_file(self, audio_path):
        """
        Transcribes a local audio file.
        Args:
            audio_path (str): The path to the local audio file.
        Returns:
            The transcription result from the server.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")
        
        print(f"Transcribing file: {audio_path}")
        # The client call will print the transcription to the console.
        # The whisper-live client doesn't return the transcription directly from the call.
        # It's printed to stdout.
        self.client(audio_path)

    def get_stream_handler(self):
        """
        Returns an instance of AudioStreamHandler to handle real-time streams.
        Returns:
            AudioStreamHandler: An instance to handle audio streams.
        """
        return self.stream_handler

if __name__ == '__main__':
    # This is an example of how to use the classes.
    # A WhisperLive server must be running.
    # To run the server:
    # pip install whisper-live[faster_whisper]
    # python -m whisper_live.server --backend faster_whisper

    # Example for file transcription
    v2t = VoiceToText()
    
    # Create a dummy audio file for testing
    try:
        import soundfile as sf
        import numpy as np
        samplerate = 16000
        duration = 5
        frequency = 440
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
        dummy_audio_path = "dummy_audio.wav"
        sf.write(dummy_audio_path, data, samplerate)

        print("--- Transcribing local file ---")
        v2t.from_file(dummy_audio_path)
        os.remove(dummy_audio_path)

    except ImportError:
        print("Please install soundfile and numpy to create a dummy audio file for testing.")
        print("pip install soundfile numpy")
    except Exception as e:
        print(f"Could not run file transcription example: {e}")
        print("Please ensure a WhisperLive server is running.")


    # Example for stream transcription
    try:
        print("\n--- Transcribing from microphone for 10 seconds ---")
        # The from_microphone method is blocking, so you would typically
        # run it in a separate thread or process in a real application.
        # For this example, we can't run it indefinitely.
        # The whisper-live client will run until Ctrl+C is pressed.
        # We will just show how to get the handler.
        stream_handler = v2t.get_stream_handler()
        print("To start microphone transcription, call: stream_handler.from_microphone()")
        print("This will run until you press Ctrl+C.")
        # stream_handler.from_microphone() # Uncomment to run
    except Exception as e:
        print(f"Could not run stream transcription example: {e}")
        print("Please ensure a WhisperLive server is running.")
