from langchain_community.vectorstores import FAISS


class VectorStoreManager:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vector_store = None

    def create_vector_store(self, documents):
        self.vector_store = FAISS.from_documents(
            documents,
            self.embedding_model
        )

    def save_vector_store(self, path="faiss_index"):
        if self.vector_store:
            self.vector_store.save_local(path)

    def load_vector_store(self, path="faiss_index"):
        self.vector_store = FAISS.load_local(
            path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
            }
        )