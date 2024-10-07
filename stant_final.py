import google.generativeai as genai
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import io
from pydub import AudioSegment
import os 



client = ElevenLabs(
  api_key="sk_509e49afc7135595102e53d0959626d9d03f9a73e045ed0f", # Defaults to ELEVEN_API_KEY
)

audio = client.generate(
  text="selam" ,
  voice="Rachel",
  model="eleven_multilingual_v2"
)

output_path = os.path.expanduser("~/Desktop/a/hello_output.mp3")


audio_data = b"".join(audio)
audio_bytes = io.BytesIO(audio_data)
audio_segment = AudioSegment.from_file(audio_bytes)
audio_segment.export("deneme.wav", format="wav")
play(audio_bytes)



load_dotenv()
api = os.getenv("GOOGLE_API_KEY")  # Doğru çevresel değişken adı

genai.configure(api_key=api)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
text = input("GDG ON CAMPUS :")
text = text + " 20 kelimeyi geçmeyecek şekilde cevap ver"
response = model.generate_content(text)
print(response.text)


client = ElevenLabs(
  api_key="sk_509586070e2d523cd429dd91e42630304a6312312e8fdc24", # Defaults to ELEVEN_API_KEY
)

audio = client.generate(
  text=response.text ,
  voice="Chris",
  model="eleven_multilingual_v2"
)

output_path = os.path.expanduser("~/Desktop/ffplay/hello_output.mp3")


audio_data = b"".join(audio)
audio_bytes = io.BytesIO(audio_data)
audio_segment = AudioSegment.from_file(audio_bytes)
audio_segment.export("ses.wav", format="wav")




# .bat dosyasının yolunu belirtin
bat_file = "Wav2LipUs.bat"

# .bat dosyasını çalıştırın
os.system("start Wav2LipUs.bat")
