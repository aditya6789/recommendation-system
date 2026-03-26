"""Simple Streamlit UI for testing recommendation endpoints."""

import streamlit as st
import requests

API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
n = st.sidebar.slider("Top N", min_value=1, max_value=20, value=10)

st.title("Recommendation System Test UI")

tab1, tab2 = st.tabs(["User Recommendations", "Similar Items"])

with tab1:
    user_id = st.number_input("User ID", min_value=1, value=1)
    if st.button("Get Recommendations"):
        try:
            response = requests.get(f"{API_BASE}/recommend/{int(user_id)}", params={"n": n}, timeout=10)
            response.raise_for_status()
            data = response.json()
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

with tab2:
    item_id = st.number_input("Item ID", min_value=1, value=1)
    if st.button("Get Similar Items"):
        try:
            response = requests.get(f"{API_BASE}/similar/{int(item_id)}", params={"n": n}, timeout=10)
            response.raise_for_status()
            data = response.json()
            st.json(data)
        except Exception as exc:
            st.error(str(exc))
