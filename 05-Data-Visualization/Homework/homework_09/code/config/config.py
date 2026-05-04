import streamlit as st


class Config:
    API_KEYS = {"mapbox": st.secrets.get("MAP_BOX_API_KEY")}
    MAP_STYLES = {
        "Dark": "mapbox://styles/mapbox/dark-v8",
        "Light": "mapbox://styles/mapbox/light-v8",
        "Road": "mapbox://styles/mapbox/streets-v8",
        "Satellite": "mapbox://styles/mapbox/satellite-v8",
    }
