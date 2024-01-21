from langchain_community.document_loaders import WebBaseLoader

def get_documents(url:str=None):
    _l = WebBaseLoader("https://n.news.naver.com/mnews/article/003/0012317114?sid=105")
    return _l.load()
