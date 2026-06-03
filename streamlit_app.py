import streamlit as st
import pandas as pd
df = pd.read_csv("mental_health_countries.csv")

st.write("""
# My first app
Hello *world!*
""")

#Testing if data set works
st.dataframe(df, use_container_width=True)
