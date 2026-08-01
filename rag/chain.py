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

        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0.3,
        )

    def ask(self, question: str):
        docs = self.retriever.invoke(question)

        if not docs:
            return {
                "answer": "I couldn't find this information in the uploaded document.",
                "sources": [],
            }

        context = ""
        source_pages = []

        for doc in docs:
            page = doc.metadata.get("page", None)
            if page is not None:
                page_number = page + 1
                source_pages.append(page_number)
            else:
                page_number = "Unknown"

            context += f"\n=========================\nPage Number: {page_number}\n=========================\n\n{doc.page_content}\n"

        source_pages = sorted(list(set(source_pages)))
        prompt = SYSTEM_PROMPT.format(context=context)
        final_prompt = f"{prompt}\n\nQuestion:\n{question}\n\nAnswer:\n"

        # Retry loop for 429 Quota Exceeded
        for attempt in range(3):
            try:
                response = self.llm.invoke(final_prompt)
                return {
                    "answer": response.content.strip(),
                    "sources": source_pages,
                }
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < 2:
                        time.sleep(12)  # Wait 12 seconds for the rate-limit window to reset
                        continue
                raise e