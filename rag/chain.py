import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from rag.prompt import SYSTEM_PROMPT

load_dotenv()


class RAGChain:
    """
    Handles Retrieval + Gemini Response Generation.
    """

    def __init__(self, retriever):
        self.retriever = retriever

        # Priority: st.secrets (Streamlit Cloud) -> os.getenv (.env / local environment)
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        # Fixed model name from gemini-2.5-flash to gemini-2.0-flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

    def ask(self, question: str):
        # Retrieve relevant chunks
        docs = self.retriever.invoke(question)

        context = ""
        source_pages = []

        if docs:
            for doc in docs:
                page = doc.metadata.get("page", None)
                if page is not None:
                    source_pages.append(page + 1)
                    page_number = page + 1
                else:
                    page_number = "Unknown"

                context += f"""
=========================
Page Number: {page_number}
=========================

{doc.page_content}
"""

        source_pages = sorted(list(set(source_pages)))

        prompt = SYSTEM_PROMPT.format(context=context)

        final_prompt = f"""
{prompt}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(final_prompt)
        answer = response.content.strip()

        return {
            "answer": answer,
            "sources": source_pages,
        }