from pathlib import Path
from typing import Dict
import os

DEFAULT_VD_NAME = os.getenv("DEFAULT_VD_NAME")


def get_default_vsparams(doc_path: str, **kwargs) -> Dict[str, object]:
    _path = Path(doc_path)
    if not _path.name and not kwargs.get("namespace"):
        raise ValueError("Path does not exist")

    _vd_name = kwargs.get("vd_name", DEFAULT_VD_NAME)

    params = {
        "vd_name": _vd_name,
        "index_name": kwargs.get("index_name", "manuals"),
        "namespace": kwargs.get("namespace") or _path.name,
        "doc_path": _path,
        "chunk_size": kwargs.get("chunk_size", 300),
        "chunk_overlap": kwargs.get("chunk_overlap", 0),
        "persist_dir": str((Path.cwd() / "core" / "db" / _vd_name).resolve()),
    }
    return params
