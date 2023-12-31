import streamlit as st
import pandas as pd
import numpy as np
from time import sleep

st.set_page_config(page_icon="🙌", page_title="스트림릿 배포하기", layout="wide")

st.subheader("도큐먼트")

if st.button("app.py 코드 보기"):
    code = """
import streamlit as st
import pandas as pd
import numpy as np
from time import sleep

st.set_page_config(page_icon="🙌", page_title="스트림릿 배포하기", layout="wide")

st.header("환영합니다. 😂")
st.subheader("맛보기")

cols = st.columns((1, 1, 2))
cols[0].metric("10/11", "15 C", "2")
cols[0].metric("10/12", "17 C", "2 F")
cols[0].metric("10/13", "15 C", "2")
cols[1].metric("10/14", "15 C", "2")
cols[1].metric("10/15", "17 C", "2 F")
cols[1].metric("10/16", "15 C", "2")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

cols[2].line_chart(chart_data)

"""
    st.code(code, language="python")
