import streamlit
from utils import return_inactive


def attach_sidebar(st: streamlit):
    import utils.file as fileUtils

    with st.sidebar:
        with st.expander("파일 업로드"):
            upload_file = st.file_uploader(
                "Upload a document", type=["pdf"], accept_multiple_files=False
            )
            if upload_file:
                fileUtils.save_buffer(
                    save_filename=upload_file.name, buffer=upload_file.getbuffer()
                )

        _filelist = fileUtils.filelist()
        with st.expander("파일목록", expanded=len(_filelist) > 0):
            selected_file = st.radio(
                "업로드 파일", options=map(lambda x: x[0], _filelist), index=None
            )

            if selected_file:
                st.session_state.file_path = list(
                    filter(lambda x: x[0] == selected_file, _filelist)
                )[0][1]

        with st.expander("조회옵션", expanded=True):
            # st.session_state.top_k = st.text_input(label="top-k", value="1")
            st.session_state.top_k = st.slider(
                "top-k", 1, 10, on_change=return_inactive
            )

        with st.expander("사용된 토큰", expanded=True):
            if "token_usage" in st.session_state:
                for key, val in st.session_state.token_usage.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(val, float):
                        val = round(val, 7)
                    st.write(key, " : ", val)
