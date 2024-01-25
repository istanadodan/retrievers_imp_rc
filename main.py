import streamlit as st
from dotenv import load_dotenv
import logging

load_dotenv()


def return_inactive():
    return


def write_answer(answer: object, cb: object):
    if not answer:
        return
    st.subheader("답변")
    st.write(answer.get("result"))

    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("출처")
        if answer.get("source_documents", None):
            st.write(
                answer.get("source_documents")[0].page_content
                + "|page="
                + str(answer["source_documents"][0].metadata.get("page", 0))
            )
    with col2:
        for key, el in cb.__dict__.items():
            if key.startswith("_"):
                continue
            st.write(key, " : ", el)


def main():
    from langchain_community.callbacks.manager import get_openai_callback
    from service.retrieval_search import QueryType, query
    import utils.file as fileUtils

    st.header("LLM 질의하기")

    if "file_path" not in st.session_state:
        st.session_state.file_path = None

    with st.sidebar:
        with st.expander("파일 업로드"):
            upload_file = st.file_uploader(
                "Upload a document", type=["pdf"], accept_multiple_files=False
            )
            if upload_file:
                fileUtils.save_buffer(
                    save_filename=upload_file.name, buffer=upload_file.getbuffer()
                )

        _filelist = fileUtils.filelist()
        with st.expander("파일목록", expanded=len(_filelist) > 0):
            selected_file = st.radio(
                "업로드 파일", options=map(lambda x: x[0], _filelist), index=None
            )
            st.session_state.file_path = None
            if selected_file:
                st.session_state.file_path = list(
                    filter(lambda x: x[0] == selected_file, _filelist)
                )[0][1]
    # tab 작성
    with get_openai_callback() as cb:
        # 탭이 1개인 경우, with문 실행 오류
        t1, t2 = st.tabs(["mquery retieval", "parent-node retrieval"])

        file_path = st.session_state.file_path
        answer = {}

        with t1:
            user_question = st.text_input(
                "파일내용에 대해 질의해 주세요.", on_change=return_inactive, key="q1"
            )
            if st.button("실행", key="b1") and user_question:
                if file_path:
                    answer = query(
                        user_question,
                        file_path,
                        query_type=QueryType.Multi_Query,
                    )
                else:
                    st.write("파일이 선택되지 않았습니다.")

            write_answer(answer=answer, cb=cb)

        with t2:
            """
            source_documents가 추출되지 않는 문제가 있음.
            """
            user_question = st.text_input(
                "파일내용에 대해 질의해 주세요.", on_change=return_inactive, key="q2"
            )
            if st.button("실행", key="b2") and user_question:
                with st.spinner():
                    if file_path:
                        answer = query(
                            user_question,
                            file_path,
                            query_type=QueryType.Parent_Document,
                        )
                    else:
                        st.write("파일이 선택되지 않았습니다.")

            write_answer(answer=answer, cb=cb)


def setup():
    st.set_page_config(page_icon="🙌", page_title="LLM Query", layout="wide")

    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    setup()
    main()
