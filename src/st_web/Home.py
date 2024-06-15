'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-05 13:34:18
LastEditors: hibana2077 hibana2077@gmaill.com
LastEditTime: 2024-06-03 19:03:40
FilePath: \Digital-TA\src\st_web\Home.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import streamlit as st
import requests
import os

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8081")

# Request to the API
status_response = requests.get(f"{API_URL}/status/mertics")
textbook_count = status_response.json()["textbook_count"]
subject_count = status_response.json()["subject_count"]
dialogue_count = status_response.json()["dialogue_count"]

# Page Widgets

## header

st.header("數位助教平台")

## logo

_, col2_logo, _ = st.columns(3)
col2_logo.image("./assets/logo.webp", width=200)

## status mertics

st.subheader("狀態")
col1, col2, col3 = st.columns(3)
col1.metric("教科書資料數", f"{textbook_count} 本", "1 本")
col2.metric("科目數", f"{subject_count} 科", "1 科")
col3.metric("對話次數", f"{dialogue_count} 次", "4%")
st.divider()

## introduction

st.subheader("簡介")
st.write("歡迎使用我們的數位助教平台，它為教育者提供優質支援，使教學更高效。本系統旨在減輕教師的負擔，以便他們能聚焦於優化教學內容。系統個性化學習方案同時也支持學生學業的發展。我們期望通過這些工具優化教學過程，推動教育的革新。讓我們一起努力，打造教育的美好未來。")
st.divider()

## Quick Start

st.subheader("快速開始")
### upload textbooks
col1_quick_start, col2_quick_start = st.columns(2)

with col1_quick_start:
    container_upload_textbooks = st.container(border=True, height=200)
    container_upload_textbooks.subheader("上傳教科書")
    container_upload_textbooks.write("上傳教科書以便數位助教平台提供更好的支援。")
    container_upload_textbooks.link_button("Go to the upload page", "/Upload")

### chat with TA

with col2_quick_start:
    container_chat_with_ta = st.container(border=True, height=200)
    container_chat_with_ta.subheader("與助教對話")
    container_chat_with_ta.write("與助教對話以獲得更多的支援。")
    container_chat_with_ta.link_button("Go to the chat page", "/Chat")

st.divider()

## Documentation

st.subheader("說明文件")
### link to the documentation (powered by docusaurus)
#### docusaurus are not implemented yet

col1_doc, col2_doc = st.columns(2)

with col1_doc:
    container_doc_left_one = st.container(border=True)
    container_doc_left_two = st.container(border=True)
    container_doc_left_three = st.container(border=True)
    container_doc_left_one.subheader("學生使用教學")
    container_doc_left_one.write("這個文件示範了學生如何使用我們的數位助教平台。")
    container_doc_left_one.link_button("Go to the documentation", "https://docusaurus.io")
    container_doc_left_two.subheader("教師使用教學")
    container_doc_left_two.write("這個文件示範了教師如何使用我們的數位助教平台。")
    container_doc_left_two.link_button("Go to the documentation", "https://docusaurus.io")
    container_doc_left_three.subheader("管理員使用教學")
    container_doc_left_three.write("這個文件示範了管理員如何使用我們的數位助教平台。")
    container_doc_left_three.link_button("Go to the documentation", "https://docusaurus.io")

with col2_doc:
    container_doc_right_one = st.container(border=True)
    container_doc_right_two = st.container(border=True)
    container_doc_right_three = st.container(border=True)
    container_doc_right_one.subheader("故障排除")
    container_doc_right_one.write("這個文件示範了如何排除數位助教平台的故障。")
    container_doc_right_one.link_button("Go to the documentation", "https://docusaurus.io")
    container_doc_right_two.subheader("常見問題")
    container_doc_right_two.write("這個文件列出了數位助教平台的常見問題。")
    container_doc_right_two.link_button("Go to the documentation", "https://docusaurus.io")
    container_doc_right_three.subheader("開發者文檔")
    container_doc_right_three.write("這個文件幫助開發者進行本系統的維護與開發。")
    container_doc_right_three.link_button("Go to the documentation", "https://docusaurus.io")
