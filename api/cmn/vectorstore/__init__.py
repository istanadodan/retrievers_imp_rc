from langchain_core.documents import Document

class VDRetrieve(Retriever):
    name:str
    def add_documents(self, docs:list[Document]):
        pass
    def retreiver

def select_vectordb(name:str) -> VectorDbLoader:
    pass