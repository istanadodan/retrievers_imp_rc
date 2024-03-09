import streamlit as st


def main():
    from ui.sidebar import show_menubar
    from cmn.config import setup

    # 앱 설정
    setup(st)
    # 메뉴 생성
    app = _create_menu()

    # 사이드바 작성
    with st.sidebar:
        show_menubar(st, app)

    with st.container(border=True):
        app.run()


def _create_menu():
    from sub_pages import doc_search, function_call, summary
    from ui.menu import MenuUI

    app = MenuUI()
    app.add_app("문서조회", doc_search)
    app.add_app("Function-call", function_call)
    app.add_app("Summary", summary)
    return app


if __name__ == "__main__":
    main()
