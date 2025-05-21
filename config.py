import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"

# Database Configuration
CHROMA_DB_PATH = "./data/chroma_db"
SQLITE_DB_PATH = "./data/content.db"

# Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTENT_LENGTH = 50000  # Limit content size for MVP
