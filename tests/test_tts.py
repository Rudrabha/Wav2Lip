import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tts import TTSError, api_key_env, synthesize_speech


class FakeResponse(io.BytesIO):
	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.close()


class TTSTest(unittest.TestCase):
	def request_for(self, provider, **kwargs):
		self.temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(self.temp_dir.cleanup)
		output_path = os.path.join(self.temp_dir.name, "speech.audio")
		with patch("tts.urlopen", return_value=FakeResponse(b"audio-data")) as mocked_urlopen:
			result = synthesize_speech(
				provider,
				"Hello from Wav2Lip",
				output_path,
				"secret-key",
				**kwargs
			)

		with open(output_path, "rb") as audio_file:
			self.assertEqual(audio_file.read(), b"audio-data")
		self.assertEqual(result, output_path)
		self.assertEqual(mocked_urlopen.call_args[1]["timeout"], 120)
		request = mocked_urlopen.call_args[0][0]
		self.assertEqual(request.get_method(), "POST")
		self.assertEqual(request.headers["Content-type"], "application/json")
		self.assertEqual(request.headers["Accept"], "audio/*")
		return request

	def test_builds_elevenlabs_request(self):
		request = self.request_for("elevenlabs", voice="voice/id")

		self.assertIn("/voice%2Fid?output_format=mp3_44100_128", request.full_url)
		self.assertEqual(request.headers["Xi-api-key"], "secret-key")
		self.assertEqual(
			json.loads(request.data.decode("utf-8")),
			{
				"text": "Hello from Wav2Lip",
				"model_id": "eleven_multilingual_v2",
			},
		)

	def test_builds_openai_request(self):
		request = self.request_for("openai")

		self.assertEqual(request.full_url, "https://api.openai.com/v1/audio/speech")
		self.assertEqual(request.headers["Authorization"], "Bearer secret-key")
		self.assertEqual(
			json.loads(request.data.decode("utf-8")),
			{
				"input": "Hello from Wav2Lip",
				"model": "gpt-4o-mini-tts",
				"voice": "coral",
				"response_format": "mp3",
			},
		)

	def test_builds_deepgram_request(self):
		request = self.request_for("deepgram", voice="aura-2-orion-en")

		self.assertIn("model=aura-2-orion-en", request.full_url)
		self.assertIn("encoding=mp3", request.full_url)
		self.assertEqual(request.headers["Authorization"], "Token secret-key")
		self.assertEqual(
			json.loads(request.data.decode("utf-8")),
			{"text": "Hello from Wav2Lip"},
		)

	def test_rejects_missing_required_values(self):
		with self.assertRaises(ValueError):
			synthesize_speech("openai", "", "speech.mp3", "key")
		with self.assertRaises(ValueError):
			synthesize_speech("openai", "hello", "speech.mp3", "")
		with self.assertRaises(ValueError):
			synthesize_speech("elevenlabs", "hello", "speech.mp3", "key")
		with self.assertRaises(ValueError):
			synthesize_speech("unknown", "hello", "speech.mp3", "key")

	def test_maps_provider_api_key_environment_variables(self):
		self.assertEqual(api_key_env("elevenlabs"), "ELEVENLABS_API_KEY")
		self.assertEqual(api_key_env("openai"), "OPENAI_API_KEY")
		self.assertEqual(api_key_env("deepgram"), "DEEPGRAM_API_KEY")

	def test_reports_api_error_without_exposing_key(self):
		error = HTTPError(
			"https://api.openai.com/v1/audio/speech",
			401,
			"Unauthorized",
			{},
			io.BytesIO(b'{"error":"invalid key"}'),
		)
		with patch("tts.urlopen", side_effect=error):
			with self.assertRaises(TTSError) as raised:
				synthesize_speech(
					"openai", "hello", "speech.mp3", "secret-key"
				)

		self.assertIn("openai request failed with HTTP 401", str(raised.exception))
		self.assertNotIn("secret-key", str(raised.exception))


if __name__ == "__main__":
	unittest.main()
