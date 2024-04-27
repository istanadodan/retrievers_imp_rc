import streamlit as st


def run():
    from langchain_community.callbacks.manager import get_openai_callback
    from service.tools.sql_agent import query
    from ui.tasks import return_inactive

    user_question = ""
    # tab 작성
    with get_openai_callback() as cb:
        top_k = st.session_state.top_k if st.session_state.top_k else 2

        # 탭이 1개인 경우, with문 실행 오류
        tabs = st.tabs(
            [
                "sql_agent",
                "parent-node",
            ]
        )

        file_path = (
            st.session_state.file_path if "file_path" in st.session_state else None
        )
        # 결과값 초기화
        query_res = (None, None)
        with tabs[0]:
            """
            sql_agent에 retriever를 활용하여 보다 쿼리 정확성을 높임.
            """
            user_question = st.text_input(
                "파일내용에 대해 질의해 주세요.",
                on_change=return_inactive,
                key="q1",
            )
            file_path = "_"
            if st.button("실행", key="b1", type="primary") and user_question:
                query_res = query(
                    user_question,
                    # file_path,
                    # query_type=QueryType.Multi_Query,
                    k=top_k,
                )

        with tabs[1]:
            """
            출처가 추출되지 않기도 함
            * ___마이데이터 문서에서는 출처가 나오지 않았고, 자바스크립트에서는 나옴.(건수 문제?)___
            """

        if user_question and not file_path:
            st.warning("파일이 선택되지 않았습니다.")

        elif query_res[1]:
            st.session_state.conversation.append(
                dict(user=query_res[0], ai=query_res[1])
            )
            st.session_state.token_usage = cb.__dict__
            st.rerun()
