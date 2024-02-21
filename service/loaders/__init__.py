from typing import Union, List
from cmn.types.loader import LoaderType
from . import pdf_loader, text_loader, web_loader
import logging

loader_map = {
    "txt": text_loader.TxtLoader,
    "pdf": pdf_loader.PDFLoader,
}


def get_documents_from_file(files: Union[List[str], str], splitter: object = None):
    import itertools, os

    files = files if isinstance(files, list) else [files]

    output_docs = []

    group = itertools.groupby(files, lambda x: os.path.splitext(x)[-1][1:])
    for key, files in group:
        if key not in loader_map.keys():
            logging.error("처리할 수 없는 파일형식입니다.")
            continue
        loader: LoaderType = loader_map[key](files, splitter)
        read_docs = loader.get_documents()
        if read_docs:
            output_docs.extend(read_docs)

    return output_docs


def get_documents_from_urls(urls: Union[List[str], str], splitter=None):
    loader = web_loader.WebLoader(urls, splitter)
    return loader.get_documents()
