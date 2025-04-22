import os
from dotenv import load_dotenv

# Load environment variables from .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(env_path)

# Project settings
VERSION = os.getenv('VERSION', '0.2.0')
MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'large-v3')
DEVICE = os.getenv('WHISPER_DEVICE', 'cuda')
COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')
BEAM_SIZE = int(os.getenv('WHISPER_BEAM_SIZE', '5'))

# OpenAI settings
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1')
OPENAI_LARGE_CONTEXT_MODEL = os.getenv('OPENAI_LARGE_CONTEXT_MODEL', 'gpt-4.1')
OPENAI_HISTORY_MODEL = os.getenv('OPENAI_HISTORY_MODEL', 'o4-mini')
OPENAI_SPEAKER_MODEL = os.getenv('OPENAI_SPEAKER_MODEL', 'o4-mini')
TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '4800'))
MAX_INPUT_TOKENS = int(os.getenv('OPENAI_MAX_INPUT_TOKENS', '60000'))
CHARS_PER_TOKEN = int(os.getenv('OPENAI_CHARS_PER_TOKEN', '4'))
MAX_FILE_SIZE_KB = int(os.getenv('MAX_FILE_SIZE_KB', '500'))

# Diarization settings
USE_SPEAKER_DIARIZATION = os.getenv('USE_SPEAKER_DIARIZATION', 'True').lower() in ('true','1','yes')
DIARIZATION_MODEL = os.getenv('DIARIZATION_MODEL', 'pyannote/speaker-diarization-3.1')
DIARIZATION_ALTERNATIVE_MODEL = os.getenv('DIARIZATION_ALTERNATIVE_MODEL', 'pyannote/segmentation-3.0')
DIARIZATION_MIN_SPEAKERS = int(os.getenv('DIARIZATION_MIN_SPEAKERS', '1'))
DIARIZATION_MAX_SPEAKERS = int(os.getenv('DIARIZATION_MAX_SPEAKERS', '10'))

# Prompt files directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROMPTS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'prompts'))
PROMPT_CLEANUP_FILE = os.path.join(PROMPTS_DIR, 'cleanup_prompt.txt')
PROMPT_SUMMARY_FILE = os.path.join(PROMPTS_DIR, 'summary_prompt.txt')
PROMPT_BLOG_FILE = os.path.join(PROMPTS_DIR, 'blog_prompt.txt')
PROMPT_BLOG_ALT1_FILE = os.path.join(PROMPTS_DIR, 'blog_alt1_prompt.txt')
PROMPT_HISTORY_EXTRACT_FILE = os.path.join(PROMPTS_DIR, 'history_extract_prompt.txt')
PROMPT_SPEAKER_ASSIGN_FILE = os.path.join(PROMPTS_DIR, 'speaker_assignment_prompt.txt')