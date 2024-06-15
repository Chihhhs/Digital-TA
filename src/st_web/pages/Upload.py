import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8081")

st.header("Upload Textbooks")

uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
password = st.text_input(label="upload_passwd",placeholder="password")
if uploaded_file is not None: # need password validation
    # 顯示一些關於文件的基本信息
    st.write("Filename:", uploaded_file.name)
    st.write("Type:", uploaded_file.type)
    st.write(f"Size: {uploaded_file.size/1024/1024:.3f} MB")

    if st.button("Upload") and password == "admin":
        file_data = uploaded_file.getvalue()
        files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            st.success("Upload successful")
        else:
            st.error("Upload failed")