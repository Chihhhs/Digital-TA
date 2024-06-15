import streamlit as st
import requests
import os

OLLAMA_SERVER = os.getenv("OLLAMA_SERVER", "http://localhost:11434")

def get_local_models() -> list:
    response = requests.get(f"{OLLAMA_SERVER}/api/tags")
    if response.status_code == 200:
        return response.json()["models"]
    else:
        return []
    
def post_pull_model(model_name: str) -> dict:
    response = requests.post(f"{OLLAMA_SERVER}/api/pull", json={"name": model_name})
    if response.status_code == 200:
        return response.json()
    else:
        return {"success": False, "message": "Failed to pull model."}

st.header("Models")

st.markdown("## Local Models")
if st.button("Refresh"):
    local_models = get_local_models()
local_models = get_local_models()
st.table(local_models)

st.markdown("## Change Chat Model")
local_models = get_local_models()
local_models_list = [model["name"] for model in local_models]
model_name = st.selectbox("Select a model", local_models_list, index=None)
if model_name != None and st.button("Change Model"):
    if "chat_model" in st.session_state:
        del st.session_state["chat_model"]
    st.session_state["chat_model"] = model_name
    st.success(f"Model changed to {model_name}")

st.markdown("## Pull Model")
st.markdown("Pull a model from the [Ollama Model Repository](https://ollama.com/)")
text_input = st.text_input("Model Name")
passwd_input = st.text_input("Password", type="password")
if st.button("Pull Model"):
    if passwd_input == "admin":
        response = post_pull_model(text_input)
        st.json(response)