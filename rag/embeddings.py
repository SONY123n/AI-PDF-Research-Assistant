import os

from dotenv import load_dotenv
from google import genai
from langchain_core.embeddings import Embeddings

load_dotenv()


class GeminiEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings using the official Google GenAI SDK.
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        self.client = genai.Client(api_key=api_key)

        self.model = "models/gemini-embedding-001"

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )

            embeddings.append(response.embeddings[0].values)

        return embeddings

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )

        return response.embeddings[0].values