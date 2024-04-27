import streamlit as st
from ui.sidebar import show_menubar
from config import page_config, config_logger


def main():
    # 로거설정
    config_logger()
    # 타이틀, 세션변수 초기설정
    page_config()

    # 메뉴 생성
    app = create_menu()
    # 사이드바 작성
    with st.sidebar:
        show_menubar(st, app)

    with st.container(border=True):
        app.run()


def create_menu():
    from apps import doc_search, function_call, summary
    from service.tools import rebate_agents, langgraph_ex1, rebate_graph
    from service.agents import history_chat
    from ui.menu import MenuUI

    app = MenuUI()
    app.add_app("문서조회", doc_search)
    app.add_app("Function-call", function_call)
    app.add_app("Summary", summary)
    app.add_app("rebate_agents", rebate_agents)
    app.add_app("langchain_graph", rebate_graph)
    app.add_app("runnable history", history_chat)

    return app


if __name__ == "__main__":
    main()
