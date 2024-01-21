import logging
from langchain_community.callbacks.manager import get_openai_callback
import streamlit as st
from pathlib import Path


def main():
    st.set_page_config(page_icon="🙌", page_title="LLM Query", layout="wide")
    st.header("LLM 질의하기")
    st.subheader("5가지 질의 방식을 테스트한다")

    if "file_path" not in st.session_state:
        st.session_state.file_path = None

    t1, t2 = st.tabs(["mquery retieval", "multi query"])
    with t1:
        st.header("질의하기")
        user_question = st.text_input("파일내용에 대해 질의해 주세요.")

        if user_question:
            file_path = st.session_state.file_path

            if not file_path:
                st.write('파일이 선택되지 않았습니다.')
                return
            # from service.multi_query import query
            # from service.parent_document import query
            # from service.self_query import query
            # from service.time_weight import query
            from service.mqry_retrieval import query

            logging.basicConfig(level=logging.INFO)

            with get_openai_callback() as cb:
                # print(query('갤S24에 대해 알아봐줘'))
                # print(query("금융데이터 산업 개황을 설명해줘."))
                answer = query(user_question, file_path)
                st.write(answer)

    with st.sidebar:
        st.header("파일 업로드")
        upload_file = st.file_uploader(
            "Upload a document", type=["pdf"], accept_multiple_files=False
        )
        if upload_file:
            file_path = str((Path(".") / upload_file.name).resolve())
            with open(file_path, "wb") as f:
                f.write(upload_file.getbuffer())

            st.session_state.file_path = file_path


if __name__ == "__main__":
    main()
