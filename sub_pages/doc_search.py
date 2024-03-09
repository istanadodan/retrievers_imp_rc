import streamlit as st


def run():
    from langchain_community.callbacks.manager import get_openai_callback
    from service import QueryType, query, simple_query
    from cmn.tasks import return_inactive

    # tab 작성
    with get_openai_callback() as cb:
        top_k = st.session_state.top_k if st.session_state.top_k else 2

        # 탭이 1개인 경우, with문 실행 오류
        tabs = st.tabs(
            [
                "multi-query",
                "parent-node",
                "context-comprs",
                "simple queury",
                "ensembles",
                "multi-vector",
            ]
        )

        file_path = (
            st.session_state.file_path if "file_path" in st.session_state else None
        )
        # 결과값 초기화
        query_res = (None, None)
        with tabs[0]:
            user_question = st.text_input(
                "파일내용에 대해 질의해 주세요.",
                on_change=return_inactive,
                key="q1",
            )

            if st.button("실행", key="b1", type="primary") and user_question:
                query_res = query(
                    user_question,
                    file_path,
                    query_type=QueryType.Multi_Query,
                    k=top_k,
                )

        with tabs[1]:
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
                    query_res = query(
                        user_question,
                        file_path,
                        query_type=QueryType.Parent_Document,
                        k=top_k,
                    )

        with tabs[2]:
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
                    query_res = query(
                        user_question,
                        file_path,
                        query_type=QueryType.Contextual_Compression,
                        k=top_k,
                    )

        with tabs[3]:
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
                    query_res = simple_query(
                        user_question,
                    )

        with tabs[4]:
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
                    query_res = query(
                        user_question,
                        file_path,
                        query_type=QueryType.Ensembles,
                        k=top_k,
                    )

        with tabs[5]:
            """
            멀티벡터 검색 - 검색대상은 자식문서이며, 결과는 그 부모문서로 한다.
            """
            user_question = st.text_input(
                "파일내용에 대해 질의해 주세요.",
                on_change=return_inactive,
                key="q6",
            )
            if st.button("실행", key="b6") and user_question:
                with st.spinner():
                    query_res = query(
                        user_question,
                        file_path,
                        query_type=QueryType.Multi_Vector,
                        k=top_k,
                    )

        if user_question and not file_path:
            st.warning("파일이 선택되지 않았습니다.")

        elif query_res[1]:
            st.session_state.conversation.append(
                dict(user=query_res[0], ai=query_res[1], source=query_res[2])
            )
            st.session_state.token_usage = cb.__dict__
            st.rerun()
