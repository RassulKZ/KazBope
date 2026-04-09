from elevenlabs import ElevenLabs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env into os.environ

API_KEY = os.getenv("ELEVENLABS_API_KEY")   # set this as env variable

if not API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(api_key=API_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this when you go to production
    allow_methods=["*"],
    allow_headers=["*"],
)