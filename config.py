import streamlit
from dotenv import load_dotenv

load_dotenv()


def config_logger():
    import logging

    logging.basicConfig(level=logging.INFO)


def page_config(st:streamlit):
    st.set_page_config(
        page_icon="🙌",
        page_title="LLM Query",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.header("LLM 질의하기")

    if "top_k" not in st.session_state:
        st.session_state.top_k = 2

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {}
    
