from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """
    Handles loading one or more PDF files.
    """

    def __init__(self):
        pass

    def load_single_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF file.
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents

    def load_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF files and preserve original filenames in metadata.
        """
        all_documents = []

        for pdf in pdf_paths:
            try:
                loader = PyPDFLoader(pdf)
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf}: {e}")

        return all_documents

    def load_from_folder(self, folder_path: str) -> List[Document]:
        """
        Load every PDF inside a target directory.
        """
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))

        all_documents = []

        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf))
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf}: {e}")

        return all_documents