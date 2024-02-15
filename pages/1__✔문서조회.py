import streamlit as st
from dotenv import load_dotenv
from template.side_bar import attach_sidebar
from template.body import write_answer
from utils import return_inactive
import logging

load_dotenv()


def write_warning(message: str):
    st.warning(message)
    # st.stop()


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
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "multi-query",
                    "parent-node",
                    "context-comprs",
                    "simple queury",
                    "ensembles",
                ]
            )

            file_path = (
                st.session_state.file_path if "file_path" in st.session_state else None
            )
            # 결과값 초기화
            answer = {}
            user_qry = None
            with tab1:
                user_question = st.text_input(
                    "파일내용에 대해 질의해 주세요.",
                    on_change=return_inactive,
                    key="q1",
                )

                if st.button("실행", key="b1", type="primary") and user_question:
                    user_qry, answer = query(
                        user_question,
                        file_path,
                        query_type=QueryType.Multi_Query,
                        k=top_k,
                    )

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
                        user_qry, answer = query(
                            user_question,
                            file_path,
                            query_type=QueryType.Parent_Document,
                            k=top_k,
                        )

            with tab3:
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
                        user_qry, answer = query(
                            user_question,
                            file_path,
                            query_type=QueryType.Contextual_Compression,
                            k=top_k,
                        )

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
                        user_qry, answer = simple_query(
                            user_question,
                        )

            with tab5:
                """
                하이브리드 검색
                """
                user_question = st.text_input(
                    "파일내용에 대해 질의해 주세요.",
                    on_change=return_inactive,
                    key="q5",
                )
                if st.button("실행", key="b5") and user_question:
                    with st.spinner():
                        user_qry, answer = query(
                            user_question,
                            file_path,
                            query_type=QueryType.Ensembles,
                            k=top_k,
                        )

            if user_question and not file_path:
                write_warning("파일이 선택되지 않았습니다.")

            elif answer:
                st.session_state.conversation.append(dict(user=user_qry, ai=answer))
                write_answer(st, cb=cb)

    except Exception as e:
        st.session_state.conversation.append(dict(user=user_question, ai=str(e)))
        write_answer(st, user_question, dict(result=e), cb=cb)


def setup():
    st.set_page_config(page_icon="🙌", page_title="LLM Query", layout="wide")
    st.header("LLM 질의하기")

    logging.basicConfig(level=logging.INFO)

    if "top_k" not in st.session_state:
        st.session_state.top_k = 1

    if "conversation" not in st.session_state:
        st.session_state.conversation = []


if __name__ == "__main__":
    setup()
    main()
