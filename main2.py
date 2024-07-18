from api.service.simple_chat import chat
from api.service.simple_rag import rag
from pathlib import Path
from api.cmn.logger import Logger

logger = Logger.getLogger(__name__)

def main():
    # chat('무엇을 하면 가장 행복할까요?')
    file_path = Path(__file__).parent / 'data'/ 'aaa.pdf'
    answer = rag(file_path)
    logger.info(f'{file_path=}, {answer=}')
    
if __name__ == '__main__':
    main()