import os
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

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

    def ask(self, question: str):

        # Retrieve relevant chunks
        docs = self.retriever.invoke(question)

        if not docs:
            return {
                "answer": "I couldn't find this information in the uploaded document.",
                "sources": [],
            }

        # Build context with page numbers
        context = ""

        source_pages = []

        for doc in docs:

            page = doc.metadata.get("page", None)

            if page is not None:
                page_number = page + 1
                source_pages.append(page_number)
            else:
                page_number = "Unknown"

            context += f"""
=========================
Page Number: {page_number}
=========================

{doc.page_content}

"""

        source_pages = sorted(list(set(source_pages)))

        prompt = SYSTEM_PROMPT.format(
            context=context
        )

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