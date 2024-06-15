import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8081")

st.header("Embeddings")

# show the embeddings
st.subheader("Available Embeddings")
response_emb = requests.get(f"{API_URL}/embedding_count")
if response_emb.status_code == 200 and response_emb.json()["embedding_count"] > 0:
    embeddings = response_emb.json()["embedding_names"]
    for embedding in embeddings:
        st.write(embedding)
else:
    st.write("No embeddings have been uploaded yet.")

# show the files that have been uploaded
st.subheader("Uploaded Files")
response = requests.get(f"{API_URL}/file_count")
if response.status_code == 200 and response.json()["file_count"] > 0:
    files = response.json()["file_names"]
    for file in files:
        st.write(file)
else:
    files = []
    st.write("No files have been uploaded yet.")

# make new embeddings
st.subheader("Create New Embeddings")

selected_file = st.selectbox("Select a file to create embeddings from", files)
embedding_name = st.text_input("Enter a name for the embedding")
auth_password = st.text_input("Enter the authentication password", type="password")

if st.button("Create Embeddings"):
    response = requests.post(
        f"{API_URL}/create_embeddings",
        json={"file_name": selected_file, "embedding_name": embedding_name, "auth_password": auth_password},
    )
    if response.status_code == 200:
        st.success(response.json()["message"] + f" Time taken: {str(response.json()['time'])} seconds.")
    else:
        st.warning("An error occurred.")