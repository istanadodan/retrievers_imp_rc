import streamlit as st


def write_answer(answer: str, cb: object):
    if not answer:
        return

    st.session_state.token_usage = cb.__dict__

    col1, col2 = st.columns([6, 4], gap="large")
    with col1:
        st.subheader("답변")
        st.write(answer.get("result"))

    with col2:
        st.subheader("출처")

        srcs = answer.get("source_documents", [])
        for src in srcs:
            st.write(src.page_content + "| page=" + str(src.metadata.get("page", 0)))
