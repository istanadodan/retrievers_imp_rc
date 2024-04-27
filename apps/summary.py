import streamlit as st


def run():
    from langchain_community.callbacks.manager import get_openai_callback
    from service import QueryType, doc_summary, webpage_summary
    from ui.tasks import return_inactive

    user_question = ""
    # tab 작성
    with get_openai_callback() as cb:
        top_k = st.session_state.top_k if st.session_state.top_k > 2 else 5

        # 탭이 1개인 경우, with문 실행 오류
        tabs = st.tabs(["document_summary", "webpage_summary"])

        file_path = (
            st.session_state.file_path if "file_path" in st.session_state else None
        )
        # 결과값 초기화
        answer = {}
        user_qry = None
        with tabs[0]:
            user_question = st.text_input(
                "파일내용에 대해 질의해 주세요.",
                on_change=return_inactive,
                key="q1",
            )

            if st.button("실행", key="b1", type="primary") and user_question:
                with st.spinner():
                    user_qry, answer = doc_summary(
                        user_question,
                        file_path,
                        engine=QueryType.Parent_Document,
                        top_k=top_k,
                    )

        with tabs[1]:
            url = st.text_input(
                "웹페이지 URL을 입력해 주세요.",
                on_change=return_inactive,
                key="q2",
            )
            user_question = st.text_input(
                "웹페이지에 대한 요약 키워드를 입력해 주세요.",
                on_change=return_inactive,
                key="q3",
            )
            file_path = "_"
            if st.button("실행", key="b2", type="primary") and user_question:
                with st.spinner():
                    user_qry, answer = webpage_summary(
                        url,
                        user_question,
                        engine=QueryType.Parent_Document,
                        top_k=top_k,
                    )

        if user_question and not file_path:
            st.warning("파일이 선택되지 않았습니다.")

        elif answer:
            st.session_state.conversation.append(dict(user=user_qry, ai=answer))
            st.session_state.token_usage = cb.__dict__
            st.rerun()
