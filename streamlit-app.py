import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ssl
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from io import BytesIO

# Set the page title and layout
st.set_page_config(page_title="Health Disease Prediction - Final Project", layout="centered")

# Custom CSS to set Times New Roman font
st.markdown("""
    <style>
        /* Set Times New Roman font for all titles and markdown */
        body {
            font-family: 'Times New Roman', serif;
        }

        /* Title Styling */
        .title {
            font-family: 'Times New Roman', serif;
            font-size: 36px;
            color: #2C3E50;
            text-align: center;
            padding-bottom: 20px;
        }

        /* Paragraph Styling */
        .streamlit-expanderHeader, .streamlit-expanderContent {
            font-family: 'Times New Roman', serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Times New Roman', serif;
        }
    </style>
""", unsafe_allow_html=True)

# Welcome Message 
st.title("Health Disease Prediction🩺")

# Introduction
st.write("""
This project focuses on **Health Disease**, showcasing our application of modeling and simulation techniques. 
We've worked together to explore synthetic health data, perform statistical analysis, and develop predictive models to assess various health risks.

This project is the culmination of our group's dedication and effort for the **CSEC 413 Modeling and Simulation** course, and we're excited to share our work with you.
""")

# Why Health Disease Prediction
st.write("""
💡 **Why Health Disease Prediction?**  
Health diseases continue to be a major concern globally. By leveraging data modeling and simulation, our aim is to provide insights that contribute to understanding and managing health risks effectively.
""")

# Thank you message
st.write("Thank you for taking the time to view our work. Happy exploring! 🤓")
