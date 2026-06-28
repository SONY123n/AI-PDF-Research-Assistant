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
        Load a single PDF.

        Args:
            pdf_path: Path to PDF

        Returns:
            List of LangChain Documents
        """

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        return documents

    def load_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load multiple PDFs.

        Args:
            pdf_paths: List of PDF paths

        Returns:
            Combined list of Documents
        """

        all_documents = []

        for pdf in pdf_paths:

            loader = PyPDFLoader(pdf)

            docs = loader.load()

            all_documents.extend(docs)

        return all_documents

    def load_from_folder(self, folder_path: str) -> List[Document]:
        """
        Load every PDF inside a folder.

        Args:
            folder_path

        Returns:
            Combined documents
        """

        folder = Path(folder_path)

        pdf_files = folder.glob("*.pdf")

        all_documents = []

        for pdf in pdf_files:

            loader = PyPDFLoader(str(pdf))

            docs = loader.load()

            all_documents.extend(docs)

        return all_documents