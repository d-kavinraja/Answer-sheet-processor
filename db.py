# db.py
import streamlit as st
from pymongo import MongoClient

client = MongoClient(st.secrets["MONGO_URI"])
db = client["answer_sheet_extracter"]

def get_user_collection():
    return db["users"]
