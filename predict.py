import time
import os
import json
from typing import Dict
from faster_whisper import WhisperModel
from cog import BasePredictor, ConcatenateIterator, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        model_name = "medium.en"
        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16")

    def predict(
        self,
        file_url: str = Input(description="URL of the wav file to predict on", default=None),
        file_string: str = Input(description="Base64 encoded string of the wav data to predict on", default=None),
    ) -> ConcatenateIterator[Dict]:

        if file_url is None and file_string is None:
            raise ValueError("Either file_url or file_string must be provided")

        temp_audio_filename = f"temp-{time.time_ns()}.wav"

        if file_url:
            response = requests.get(file_url)
            with open(temp_audio_filename, 'wb') as file:
                file.write(response.content)
        elif file_string:
            wav_data = base64.b64decode(file_string)
            with open(temp_audio_filename, 'wb') as file:
                file.write(wav_data)

        # Transcribe audio
        print("Starting transcribing")
        options = dict(vad_filter=True)
        segments, transcript_info = self.model.transcribe(temp_audio_filename, **options)
        for s in segments:
            yield dict(start=s.start, end=s.end, text=s.text)

        # Delete temp file
        os.remove(temp_audio_filename)
