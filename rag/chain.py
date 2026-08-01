import os
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.prompt import SYSTEM_PROMPT

load_dotenv()


class RAGChain:

    def __init__(self, retriever):
        self.retriever = retriever

        api_key = (
            st.secrets.get("GOOGLE_API_KEY")
            if "GOOGLE_API_KEY" in st.secrets
            else os.getenv("GOOGLE_API_KEY")
        )

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

    def ask(self, question):
        docs = self.retriever.invoke(question)

        if not docs:
            return {
                "answer": "I couldn't find this information in the uploaded document.",
                "sources": [],
            }

        context = ""
        pages = set()

        for doc in docs:
            page = doc.metadata.get("page")

            if page is not None:
                pages.add(page + 1)

            context += f"\n{doc.page_content}\n"

        prompt = SYSTEM_PROMPT.format(context=context)

        final_prompt = f"""
{prompt}

Question:
{question}

Answer:
"""

        for attempt in range(3):
            try:
                response = self.llm.invoke(final_prompt)

                return {
                    "answer": response.content.strip(),
                    "sources": sorted(pages),
                }

            except Exception as e:
                if (
                    "429" in str(e)
                    or "RESOURCE_EXHAUSTED" in str(e)
                ) and attempt < 2:
                    time.sleep(12)
                    continue

                raise