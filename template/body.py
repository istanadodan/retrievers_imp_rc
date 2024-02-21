import streamlit as st
from template.htmlTemplates import css, bot_template, user_template
import logging


def write_answer(cb: object):
    st.balloons()
    st.markdown(css, unsafe_allow_html=True)

    st.session_state.token_usage = cb.__dict__

    with st.container(border=True):
        # with st.empty():
        #     for seconds in range(10):
        #         st.write(f"⏳ {seconds} seconds have passed")
        #         time.sleep(1)
        #     st.write("✔️ 10 seconds over!")

        # st.session_state.token_usage = cb.__dict__
        with st.expander("출처보기", expanded=False):
            srcs = st.session_state.conversation[-1].get("source", [])
            for src in srcs:
                st.write(
                    insert_line_feed(
                        src.page_content
                        + " | page="
                        + str(int(src.metadata.get("page", 0)))
                    ),
                )

        result_tab, token_tab = st.columns([7.5, 2.5], gap="medium")

        with result_tab:
            for c in reversed(st.session_state.conversation):
                if c.get("user"):
                    st.markdown(
                        user_template.format(MSG=c.get("user")),
                        unsafe_allow_html=True,
                    )
                if c.get("ai"):
                    st.markdown(
                        bot_template.format(MSG=insert_line_feed(c.get("ai"))),
                        unsafe_allow_html=True,
                    )
            # st.markdown(user_template.format(MSG=query), unsafe_allow_html=True)
            # st.markdown(
            #     bot_template.format(MSG=insert_line_feed(answer.get("result"))),
            #     unsafe_allow_html=True,
            # )
            # st.text_area(label="답변", value=insert_line_feed(answer.get("result")))

        with token_tab:
            with st.container(border=True):
                st.markdown(
                    f"<span style='color:blue;background-color:yellow'>마지막 사용 토큰</span>",
                    unsafe_allow_html=True,
                )
                for key, val in st.session_state.token_usage.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(val, float):
                        val = round(val, 7)
                    st.write(key, " : ", val)

        # st.rerun()


def insert_line_feed(txt: str, match: str = "다\.", rep: str = "\n"):
    import re

    if not isinstance(txt, str):
        raise Exception(txt)
    return re.sub(f"({match})", rf"\1{rep}", txt)


def setup():

    st.set_page_config(page_icon="🙌", page_title="LLM Query", layout="wide")
    st.header("LLM 질의하기")

    logging.basicConfig(level=logging.INFO)

    if "top_k" not in st.session_state:
        st.session_state.top_k = 1

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {}
