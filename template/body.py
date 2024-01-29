import streamlit as st


def write_answer(answer: str, cb: object):
    if not answer:
        return

    st.session_state.token_usage = cb.__dict__

    col1, _ = st.columns([9, 1], gap="large")
    with col1:
        st.subheader("답변")
        st.write(insert_line_feed(answer.get("result")))

    with st.expander("출처 보기", expanded=False):
        srcs = answer.get("source_documents", [])
        for src in srcs:
            st.write(
                insert_line_feed(
                    src.page_content + "| page=" + str(src.metadata.get("page", 0))
                )
            )


def insert_line_feed(txt: str, match: str = "다\.", rep: str = "\n"):
    import re

    return re.sub(f"({match})", rf"\1{rep}", txt)
