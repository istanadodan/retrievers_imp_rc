import streamlit as st
from dotenv import load_dotenv
import logging
from template.side_bar import attach_sidebar
from template.body import write_answer
from utils import return_inactive

load_dotenv()


def write_warning(message: str):
    st.warning(message)
    # st.stop()


# def write_answer(answer: str, cb: object):
#     if not answer:
#         return

#     st.session_state.token_usage = cb.__dict__

#     col1, col2 = st.columns([6, 4], gap="large")
#     with col1:
#         st.subheader("답변")
#         st.write(answer.get("result"))

#     with col2:
#         st.subheader("출처")
#         if answer.get("source_documents", None):
#             st.write(
#                 answer.get("source_documents")[0].page_content
#                 + "| page="
#                 + str(answer["source_documents"][0].metadata.get("page", 0))
#             )


def main():
    from langchain_community.callbacks.manager import get_openai_callback
    from service import QueryType, query, simple_query

    # 사이드바 작성
    attach_sidebar(st)

    # tab 작성
    try:
        with get_openai_callback() as cb:
            top_k = st.session_state.top_k if st.session_state.top_k else 2

            # 탭이 1개인 경우, with문 실행 오류
            tab1, tab2, t3, tab4 = st.tabs(
                ["multi-query", "parent-node", "context-comprs", "simple queury"]
            )

            file_path = (
                st.session_state.file_path if "file_path" in st.session_state else None
            )
            answer = {}

            with tab1:
                user_question = st.text_input(
                    "파일내용에 대해 질의해 주세요.",
                    on_change=return_inactive,
                    key="q1",
                )
                if st.button("실행", key="b1", type="primary") and user_question:
                    if file_path:
                        answer = query(
                            user_question,
                            file_path,
                            query_type=QueryType.Multi_Query,
                            k=top_k,
                        )

                        write_answer(answer=answer, cb=cb)
                    else:
                        write_warning("파일이 선택되지 않았습니다.")

            with tab2:
                """
                출처가 추출되지 않기도 함
                * ___마이데이터 문서에서는 출처가 나오지 않았고, 자바스크립트에서는 나옴.(건수 문제?)___
                """
                user_question = st.text_input(
                    "파일내용에 대해 질의해 주세요.",
                    on_change=return_inactive,
                    key="q2",
                )
                if st.button("실행", key="b2", type="secondary") and user_question:
                    with st.spinner():
                        if file_path:
                            answer = query(
                                user_question,
                                file_path,
                                query_type=QueryType.Parent_Document,
                                k=top_k,
                            )
                            write_answer(answer=answer, cb=cb)
                        else:
                            write_warning("파일이 선택되지 않았습니다.")

            with t3:
                """
                조회결과를 LLM으로 압축하거나 조회결과의 필터링을 한다.
                """
                user_question = st.text_input(
                    "파일내용에 대해 질의해 주세요.",
                    on_change=return_inactive,
                    key="q3",
                )
                if st.button("실행", key="b3") and user_question:
                    with st.spinner():
                        if file_path:
                            answer = query(
                                user_question,
                                file_path,
                                query_type=QueryType.Contextual_Compression,
                                k=top_k,
                            )

                            write_answer(answer=answer, cb=cb)
                        else:
                            write_warning("파일이 선택되지 않았습니다.")

            with tab4:
                """
                단순 조회
                """
                user_question = st.text_input(
                    "파일내용에 대해 질의해 주세요.",
                    on_change=return_inactive,
                    key="q4",
                )
                if st.button("실행", key="b4") and user_question:
                    with st.spinner():
                        answer = simple_query(
                            user_question,
                        )

                        write_answer(answer=answer, cb=cb)

    except Exception as e:
        write_answer(answer=dict(result=e), cb=cb)


def setup():
    st.set_page_config(page_icon="🙌", page_title="LLM Query", layout="wide")
    st.header("LLM 질의하기")

    logging.basicConfig(level=logging.INFO)

    if "top_k" not in st.session_state:
        st.session_state.top_k = 1


if __name__ == "__main__":
    setup()
    main()
