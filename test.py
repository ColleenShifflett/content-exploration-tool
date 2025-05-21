import os
from dotenv import load_dotenv

load_dotenv()

def test_environment():
    """Test that environment variables are loaded correctly."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key_preview = api_key[:3] + "..." + api_key[-3:] if len(api_key) > 6 else "***"
        print(f"✅ OpenAI API key found: {api_key_preview}")
    else:
        print("❌ OpenAI API key not found")

def test_imports():
    """Test that all required packages are installed."""
    try:
        import streamlit
        import langchain
        import chromadb
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")

if __name__ == "__main__":
    print("Running environment tests...")
    test_environment()
    test_imports()