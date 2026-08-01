import os
import streamlit as st

from dotenv import load_dotenv
from google import genai
from langchain_core.embeddings import Embeddings


load_dotenv()


class GeminiEmbeddings(Embeddings):


    def __init__(self):

        api_key = None


        # Streamlit Cloud secrets
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")

        except Exception:
            pass


        # Local .env fallback
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")


        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found."
            )


        self.client = genai.Client(
            api_key=api_key
        )


        self.model = "gemini-embedding-001"



    def embed_documents(self, texts):

        if not texts:
            return []


        response = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )


        return [
            embedding.values
            for embedding in response.embeddings
        ]



    def embed_query(self, text):

        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )


        return response.embeddings[0].values