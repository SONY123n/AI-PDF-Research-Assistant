import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_core.embeddings import Embeddings

load_dotenv()


class GeminiEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings using the official Google GenAI SDK.
    """

    def __init__(self):
        # Priority: st.secrets -> os.getenv
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        self.client = genai.Client(api_key=api_key)
        self.model = "text-embedding-004"  # Recommended embedding model name

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            if not text or not text.strip():
                continue
                
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