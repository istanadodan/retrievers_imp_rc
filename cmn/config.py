import logging
from dotenv import load_dotenv

load_dotenv()


def setup(st: any):
    st.set_page_config(
        page_icon="🙌",
        page_title="LLM Query",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.header("LLM 질의하기")

    logging.basicConfig(level=logging.INFO)

    if "top_k" not in st.session_state:
        st.session_state.top_k = 2

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {}
