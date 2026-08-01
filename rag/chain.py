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

        api_key = None

        # Streamlit Cloud Secret
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass

        # Local .env fallback
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")


        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
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

            page = doc.metadata.get("page")


            if page is not None:
                page_number = page + 1
                source_pages.append(page_number)

            else:
                page_number = "Unknown"


            context += (
                f"\n=========================\n"
                f"Page Number: {page_number}\n"
                f"=========================\n\n"
                f"{doc.page_content}\n"
            )


        source_pages = sorted(set(source_pages))


        prompt = SYSTEM_PROMPT.format(
            context=context
        )


        final_prompt = f"""
{prompt}

Question:
{question}

Answer:
"""


        for attempt in range(3):

            try:

                response = self.llm.invoke(final_prompt)


                content = response.content


                # Gemini can return string or list
                if isinstance(content, str):

                    answer = content


                elif isinstance(content, list):

                    answer_parts = []

                    for item in content:

                        if isinstance(item, str):
                            answer_parts.append(item)


                        elif isinstance(item, dict):

                            text = item.get("text")

                            if text:
                                answer_parts.append(text)


                        elif hasattr(item, "text"):

                            answer_parts.append(item.text)


                    answer = "\n".join(answer_parts)


                else:

                    answer = str(content)



                answer = answer.strip()


                return {
                    "answer": answer,
                    "sources": source_pages,
                }


            except Exception as e:


                error = str(e)


                if (
                    ("429" in error or "RESOURCE_EXHAUSTED" in error)
                    and attempt < 2
                ):

                    time.sleep(30)
                    continue


                raise e