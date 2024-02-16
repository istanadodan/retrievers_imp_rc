from typing import Union, List
from . import pdf_loader, text_loader, web_loader
import logging


def get_documents(files: Union[List[str], str]):
    import itertools, os

    files = files if isinstance(files, list) else [files]

    docs = []
    group = itertools.groupby(files, lambda x: os.path.splitext(x)[-1][1:])
    for key, files in group:
        loader = None
        if key == "pdf":
            loader = pdf_loader.get_loader(files)
        elif key == "txt":
            loader = text_loader.get_loader(files)
        else:
            logging.error("처리할 수 없는 파일형식입니다.")

        if loader:
            docs.extend(loader.get_documents())

    return docs


def get_documents_from_urls(urls: Union[List[str], str]):
    loader = web_loader.get_loader(urls)
    return loader.get_documents()
