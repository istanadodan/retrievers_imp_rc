import streamlit
from utils import return_inactive
import pandas as pd


def attach_sidebar(st: streamlit):
    import utils.file as fileUtils
    import time

    _filelist = fileUtils.filelist()

    with st.sidebar:
        with st.expander("파일 업로드"):
            upload_file = st.file_uploader(
                "Upload a document", type=["pdf"], accept_multiple_files=False
            )
            columns = st.columns([5.5, 4.5], gap="large")
            with columns[0]:
                is_pd_retriever = st.checkbox("PD탐색기")

            with columns[1]:
                if st.button("올리기") and upload_file:
                    from service import persist_to_vectorstore

                    persist_path = fileUtils.save_buffer(
                        save_filename=upload_file.name,
                        buffer=upload_file.getbuffer(),
                    )
                    # vectorstore에 저장한다.
                    persist_to_vectorstore(persist_path, is_pd_retriever)

        # with st.expander("데이터 업로드"):
        # data_df = pd.DataFrame({'index':['a','b'], 'data':[True, False]})
        #     st.data_editor(
        #         data_df,
        #         column_config={
        #             "index": st.column_config.TextColumn("라벨"),
        #             "data": st.column_config.CheckboxColumn(
        #                 "선택", width="medium", default=False
        #             ),
        #         },
        #         disabled=["index"],
        #         hide_index=True,
        #     )

        with st.expander("파일목록", expanded=len(_filelist) > 0):
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

        with st.expander("옵션", expanded=False):
            st.session_state.top_k = st.slider(
                "top-k", 1, 10, on_change=return_inactive
            )
