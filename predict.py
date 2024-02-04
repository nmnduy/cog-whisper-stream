import time
import os
import json
import requests
import base64
from typing import Dict
from faster_whisper import WhisperModel
from cog import BasePredictor, ConcatenateIterator, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        model_name = "medium.en"
        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16"
        )

    def predict(
        self,
        file_url: str = Input(description="URL of the wav file to predict on", default=None),
        wav_b64: str = Input(description="Base64 encoded string of the wav data to predict on", default=None),
    ) -> ConcatenateIterator[str]:

        if file_url is None and wav_b64 is None:
            raise ValueError("Either file_url or wav_b64 must be provided")

        temp_audio_filename = f"temp-{time.time_ns()}.wav"

        if file_url:
            response = requests.get(file_url)
            with open(temp_audio_filename, 'wb') as file:
                file.write(response.content)
        elif wav_b64:
            wav_data = base64.b64decode(wav_b64)
            with open(temp_audio_filename, 'wb') as file:
                file.write(wav_data)

        # Transcribe audio
        print("Starting transcribing")
        options = dict(vad_filter=True)
        segments, transcript_info = self.model.transcribe(temp_audio_filename, **options)
        for s in segments:
            yield json.dumps(dict(start=s.start, end=s.end, text=s.text, done=False))

        yield json.dumps(dict(start=0, end=0, text="", done=True))
        # Delete temp file
        os.remove(temp_audio_filename)
