from pathlib import Path
from typing import Dict
import os

DEFAULT_VD_NAME = os.getenv("DEFAULT_VD_NAME")


def get_default_vsparams(doc_path: str, **kwargs) -> Dict[str, object]:
    _path = Path(doc_path)
    if not _path.name or not _path.exists():
        raise ValueError("Path does not exist")

    params = {
        "vd_name": kwargs.get("vd_name", DEFAULT_VD_NAME),
        "index_name": kwargs.get("index_name", "manuals"),
        "namespace": _path.name,
        "doc_path": str(_path.resolve()),
        "chunk_size": kwargs.get("chunk_size", 300),
        "chunk_overlap": kwargs.get("chunk_overlap", 0),
    }
    return params
