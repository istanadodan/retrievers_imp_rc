import logging
from langchain_community.callbacks.manager import get_openai_callback

def main():
    # from service.multi_query import query
    # from service.parent_document import query
    # from service.self_query import query
    # from service.time_weight import query
    from service.retrievalqa import query
    logging.basicConfig(level=logging.INFO)
    
    with get_openai_callback() as cb:
        # print(query('갤S24에 대해 알아봐줘'))
        print(query('금융데이터 산업 개황을 설명해줘.'))
        print(cb)

if __name__=='__main__':
    main()