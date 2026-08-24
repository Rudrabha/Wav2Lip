"""Dependency-free text-to-speech providers for Wav2Lip inference."""

from __future__ import print_function

import json
import os
from shutil import copyfileobj
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


PROVIDERS = ("elevenlabs", "openai", "deepgram")

_PROVIDER_CONFIG = {
	"elevenlabs": {
		"api_key_env": "ELEVENLABS_API_KEY",
		"api_url": "https://api.elevenlabs.io/v1/text-to-speech",
		"model": "eleven_multilingual_v2",
		"output_format": "mp3_44100_128",
	},
	"openai": {
		"api_key_env": "OPENAI_API_KEY",
		"api_url": "https://api.openai.com/v1/audio/speech",
		"voice": "coral",
		"model": "gpt-4o-mini-tts",
		"output_format": "mp3",
	},
	"deepgram": {
		"api_key_env": "DEEPGRAM_API_KEY",
		"api_url": "https://api.deepgram.com/v1/speak",
		"model": "aura-2-thalia-en",
		"output_format": "mp3",
	},
}


class TTSError(RuntimeError):
	"""Raised when a text-to-speech provider cannot generate speech."""


def api_key_env(provider):
	"""Return the environment variable used for a provider API key."""
	_validate_provider(provider)
	return _PROVIDER_CONFIG[provider]["api_key_env"]


def synthesize_speech(provider, text, output_path, api_key, voice=None,
					  model=None, output_format=None, timeout=120, api_url=None):
	"""Generate speech with a configured provider and save the audio.

	Provider-native model, voice, and output-format names are accepted. Defaults
	are selected per provider. ElevenLabs requires an explicit voice ID; OpenAI
	defaults to ``coral``; Deepgram includes the voice in its model name.
	"""
	_validate_provider(provider)
	if not text or not text.strip():
		raise ValueError("TTS text cannot be empty")
	if not api_key or not api_key.strip():
		raise ValueError("{} API key cannot be empty".format(provider))

	config = _PROVIDER_CONFIG[provider]
	if provider == "deepgram" and voice and not model:
		# Deepgram encodes the voice in the model name (for example,
		# aura-2-thalia-en), so accept either shared CLI option.
		model = voice
	model = model or config["model"]
	voice = voice or config.get("voice")
	output_format = output_format or config["output_format"]
	api_url = api_url or config["api_url"]

	if provider == "elevenlabs":
		if not voice or not voice.strip():
			raise ValueError("ElevenLabs requires a voice ID")
		request = _elevenlabs_request(
			text, voice, model, output_format, api_key, api_url
		)
	elif provider == "openai":
		request = _openai_request(
			text, voice, model, output_format, api_key, api_url
		)
	else:
		request = _deepgram_request(
			text, model, output_format, api_key, api_url
		)

	try:
		with urlopen(request, timeout=timeout) as response:
			output_dir = os.path.dirname(os.path.abspath(output_path))
			if not os.path.isdir(output_dir):
				os.makedirs(output_dir)
			with open(output_path, "wb") as audio_file:
				copyfileobj(response, audio_file)
	except HTTPError as error:
		try:
			details = error.read().decode("utf-8", "replace")[:500]
		finally:
			error.close()
		raise TTSError(
			"{} request failed with HTTP {}: {}".format(
				provider, error.code, details or error.reason
			)
		)
	except URLError as error:
		raise TTSError(
			"Could not connect to {}: {}".format(provider, error.reason)
		)

	return output_path


def _validate_provider(provider):
	if provider not in PROVIDERS:
		raise ValueError(
			"Unknown TTS provider {!r}; choose from {}".format(
				provider, ", ".join(PROVIDERS)
			)
		)


def _json_request(url, payload, headers):
	request_headers = {"Accept": "audio/*", "Content-Type": "application/json"}
	request_headers.update(headers)
	return Request(
		url,
		data=json.dumps(payload).encode("utf-8"),
		headers=request_headers,
		method="POST",
	)


def _elevenlabs_request(text, voice, model, output_format, api_key, api_url):
	url = "{}/{}?{}".format(
		api_url.rstrip("/"),
		quote(voice.strip(), safe=""),
		urlencode({"output_format": output_format}),
	)
	return _json_request(
		url,
		{"text": text, "model_id": model},
		{"xi-api-key": api_key},
	)


def _openai_request(text, voice, model, output_format, api_key, api_url):
	return _json_request(
		api_url,
		{
			"input": text,
			"model": model,
			"voice": voice,
			"response_format": output_format,
		},
		{"Authorization": "Bearer {}".format(api_key)},
	)


def _deepgram_request(text, model, output_format, api_key, api_url):
	url = "{}?{}".format(
		api_url,
		urlencode({"model": model, "encoding": output_format}),
	)
	return _json_request(
		url,
		{"text": text},
		{"Authorization": "Token {}".format(api_key)},
	)
