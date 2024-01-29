from typing import Dict
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        model_name = "medium.en"
        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16")

    def predict(
        self,
        file_url: str = Input(description="URL of the wav file to predict on"),
    ) -> ConcatenateIterator[str]:
        response = requests.get(file_url)
        temp_audio_filename = f"temp-{time.time_ns()}.wav"
        with open(temp_audio_filename, 'wb') as file:
            file.write(response.content)

        # Transcribe audio
        print("Starting transcribing")
        options = dict(vad_filter=True)
        segments, transcript_info = self.model.transcribe(temp_audio_filename, **options)
        for s in segments:
            yield json.dumps(s)

        # Delete temp file
        os.remove(temp_audio_filename)
