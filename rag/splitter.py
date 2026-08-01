from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """
    Splits LangChain Documents into smaller chunks for vector embeddings.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Splits documents and strips empty chunks to avoid embedding errors.
        """
        raw_chunks = self.splitter.split_documents(documents)
        
        # Filter out empty or whitespace-only chunks
        clean_chunks = [
            chunk for chunk in raw_chunks 
            if chunk.page_content and chunk.page_content.strip()
        ]

        return clean_chunks

    def print_statistics(
        self,
        chunks: List[Document]
    ):
        print("=" * 50)
        print("Chunk Statistics")
        print("=" * 50)
        print(f"Total Chunks : {len(chunks)}")

        if chunks:
            print(f"First Chunk Length : {len(chunks[0].page_content)}")
            print(f"Last Chunk Length  : {len(chunks[-1].page_content)}")

        print("=" * 50)