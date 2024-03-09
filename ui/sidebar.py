import streamlit
from ui.menu import MenuUI


def show_menubar(st: streamlit, menu: MenuUI) -> None:
    import cmn.utils.file as fileUtils
    from cmn.tasks import return_inactive

    _filelist = fileUtils.filelist()

    menu.create(default_index=0)

    with st.expander("파일목록", expanded=False):
        selected_file = st.radio(
            "FILE",
            options=map(lambda x: x[0], _filelist),
            index=None,
            on_change=return_inactive,
        )

        if selected_file:
            st.session_state.file_path = list(
                filter(lambda x: x[0] == selected_file, _filelist)
            )[0][1]

    with st.expander("옵션", expanded=True):
        st.session_state.top_k = st.slider("top-k", 1, 10, on_change=return_inactive)

    with st.expander("토큰량", expanded=(st.session_state.token_usage != None)):
        # st.markdown(
        #     f"<span style='color:blue;background-color:yellow'>마지막 사용 토큰</span>",
        #     unsafe_allow_html=True,
        # )
        for key, val in st.session_state.token_usage.items():
            if key.startswith("_"):
                continue
            if isinstance(val, float):
                val = round(val, 7)
            st.write(key, " : ", val)

    with st.expander("파일 업로드"):
        upload_file = st.file_uploader(
            "Upload a document", type=["pdf", "txt"], accept_multiple_files=False
        )
        columns = st.columns([5.5, 4.5], gap="large")
        with columns[0]:
            is_pd_retriever = st.checkbox("PD탐색기")

        with columns[1]:
            if st.button("올리기") and upload_file:
                from service import persist_to_vectorstore

                with st.spinner():
                    persist_path = fileUtils.save_buffer(
                        save_filename=upload_file.name,
                        buffer=upload_file.getbuffer(),
                    )
                    # vectorstore에 저장한다.
                    persist_to_vectorstore(persist_path, is_pd_retriever)
                    # 재시동 - 파일명 출력
                    st.rerun()
