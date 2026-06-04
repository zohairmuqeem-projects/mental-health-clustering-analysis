import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
df = pd.read_csv("mental_health_countries.csv")

st.write("""
# 2. Dataset Discussion
The data for the analysis was retrieved via Kaggle from user Amit. In our analysis, we specifically utilized a subset of the "Uncover Global Trends in Mental Health Disorder" dataset which is restricted to the year 2017. While sourced through Kaggle, the original dataset Our World in Data is compiled from a separate source and associated blog. Links below have been provided to the Kaggle Source, Primary Source, and Amit's data.world profile.

The following links were accessed and the associated dataset was downloaded on May 12th, 2026.

Kaggle Source: https://www.kaggle.com/datasets/thedevastator/uncover-global-trends-in-mental-health-disorder/data

Primary Source: https://ourworldindata.org and Blog Post: https://ourworldindata.org/mental-health#all-charts-preview

Amit's data.hub profile: https://data.world/amitd?preview=vizzup%2Fmental-health-depression-disorder-data
""")

#Codeblock
code = """import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

df = pd.read_csv('mental_health_countries.csv')
df.head()"""
#End Codeblock

#Display code function
st.code(code, language="python")
#Testing if data set works
st.dataframe(df, use_container_width=True)

st.write("""
# 3. Dataset Cleaning and Exploration
         
         """)

