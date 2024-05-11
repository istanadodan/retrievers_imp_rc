from streamlit_option_menu import option_menu


class MenuUI:
    def __init__(self, title: str = "Main"):
        self.apps = []
        self.current_app = None
        self.main_menu = title

    def add_app(self, title, module):
        self.apps.append({"title": title, "module": module})

    def create(self, default_index: int = 0):
        self.current_app = option_menu(
            menu_title=self.main_menu,
            options=list(map(lambda x: x["title"], self.apps)),
            icons=["house", "gear"],  # person-circle, trophy-fill, chat-fill
            menu_icon="chat-text-fill",
            default_index=default_index,
            styles={
                "container": {"padding": "5!important", "background-color": "grey"},
                "icon": {"color": "white", "font-size": "15px"},
                "nav-link": {
                    "font-size": "15px",
                    "color": "blue",
                    "text-align": "left",
                    "margin": "1px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#02ab21"},
            },
        )

    def run(self):
        import logging
        from frontend.tasks import write_answer

        if not self.current_app:
            return
        app = list(filter(lambda x: x["title"] == self.current_app, self.apps))[0]

        try:
            app["module"].run()

            write_answer()

        except Exception as e:
            logging.error(f"err: {e}")

        return app["title"]
