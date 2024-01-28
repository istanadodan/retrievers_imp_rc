import streamlit


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

        with st.expander("사용된 토큰", expanded=True):
            if "token_usage" in st.session_state:
                for key, el in st.session_state.token_usage.items():
                    if key.startswith("_"):
                        continue
                    st.write(key, " : ", el)
