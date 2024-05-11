import pathlib as lib
import logging
from typing import List

PUBLIC_DIR = lib.Path("./assets/download_docs/")
if not PUBLIC_DIR.exists():
    import os
    os.makedirs(PUBLIC_DIR.resolve(), exist_ok=True)

# 스트림을 저장하고 저장위치를 반환한다.
def save_buffer(save_filename: str, buffer: memoryview) -> str:
    if buffer:        
        save_file_path = str((PUBLIC_DIR / save_filename).resolve())        
        with open(save_file_path, "wb") as f:
            f.write(buffer)

        logging.info(f"파일 저장완료:{save_file_path}")
        return save_file_path
    else:
        raise ValueError("빈 파일을 입력하였습니다.")


def filelist(file_type: List[str] = ["pdf", "txt"]) -> List[str]:
    return [
        (file.name, str(file.resolve()))
        for file in PUBLIC_DIR.glob("*.*")
        if file.suffix[1:].lower() in file_type
    ]
