import streamlit as st
from dotenv import load_dotenv
import logging
from template.side_bar import attach_sidebar
from template.body import write_answer

load_dotenv()


def return_inactive():
    return


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
    global g_answer

    from langchain_community.callbacks.manager import get_openai_callback
    from service import QueryType, query

    st.header("LLM 질의하기")

    # tab 작성
    try:
        with get_openai_callback() as cb:
            # 탭이 1개인 경우, with문 실행 오류
            t1, t2, t3 = st.tabs(["multi-query", "parent-node", "context-comprs"])

            file_path = (
                st.session_state.file_path if "file_path" in st.session_state else None
            )
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

                        write_answer(answer=answer, cb=cb)
                    else:
                        st.write("파일이 선택되지 않았습니다.")

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

                            write_answer(answer=answer, cb=cb)
                        else:
                            st.write("파일이 선택되지 않았습니다.")

            with t3:
                """
                조회결과를 LLM으로 압축하거나 조회결과의 필터링을 한다.
                """
                user_question = st.text_input(
                    "파일내용에 대해 질의해 주세요.", on_change=return_inactive, key="q3"
                )
                if st.button("실행", key="b3") and user_question:
                    with st.spinner():
                        if file_path:
                            answer = query(
                                user_question,
                                file_path,
                                query_type=QueryType.Contextual_Compression,
                            )

                            write_answer(answer=answer, cb=cb)
                        else:
                            st.write("파일이 선택되지 않았습니다.")

    except Exception as e:
        write_answer(answer=dict(result=e), cb=cb)

    attach_sidebar(st)


def setup():
    st.set_page_config(page_icon="🙌", page_title="LLM Query", layout="wide")

    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    setup()
    main()
