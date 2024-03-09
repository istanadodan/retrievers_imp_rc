from cmn.config import setup, logging, st
from ui.menu import MenuUI


def main():
    from ui.sidebar import show_menubar
    from sub_pages import doc_search, function_call, summary

    app = MenuUI()
    app.add_app("문서조회", doc_search)
    app.add_app("Function-call", function_call)
    app.add_app("Summary", summary)

    # 앱설정
    setup()
    # 사이드바 작성
    with st.sidebar:
        show_menubar(st, app)

    with st.container(border=True):
        app.run()


if __name__ == "__main__":
    main()
