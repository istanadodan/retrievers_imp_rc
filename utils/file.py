import pathlib as lib
import logging
from typing import List

PUBLIC_DIR = "./public"


def save_buffer(save_filename: str, buffer: memoryview) -> None:
    if buffer:
        save_file_path = str((lib.Path(PUBLIC_DIR) / save_filename).resolve())
        with open(save_file_path, "wb") as f:
            f.write(buffer)

        logging.info(f"파일 저장완료:{save_file_path}")
    else:
        raise ValueError("빈 파일을 입력하였습니다.")


def filelist(file_type: str = "pdf") -> List[str]:
    return [
        (file.name, str(file.resolve()))
        for file in lib.Path(PUBLIC_DIR).glob(f"*.{file_type}")
    ]
