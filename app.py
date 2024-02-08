import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from model_train_evaluate import show_model_train_page

page = st.sidebar.selectbox("Options",
                            ("Predict", "Explore", "Train/Evaluate"))

if page == "Predict":
    show_predict_page()
if page == "Train/Evaluate":
    show_model_train_page()
if page == "Explore":
    show_explore_page()
