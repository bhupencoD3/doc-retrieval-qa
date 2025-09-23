import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Updated import for OpenAI models

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "gpt-4o-mini"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    @classmethod
    def get_llm(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return ChatOpenAI(model=cls.LLM_MODEL, api_key=cls.OPENAI_API_KEY)