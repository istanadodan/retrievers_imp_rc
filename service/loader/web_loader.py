from langchain_community.document_loaders import WebBaseLoader
from service.utils.text_split import get_splitter


def get_documents(url: str = None):

    _l = WebBaseLoader(url)
    return _l.load_and_split(text_splitter=get_splitter())
