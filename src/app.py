import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import os
from pathlib import Path
import streamlit as st
from src.agent import RAGAgent

st.set_page_config(
    page_title="Agentic Memory RAG Platform",
    page_icon="⚡",
    layout="wide"
)

st.title("🪲 RoachMind: Persistent Agentic Memory RAG")
st.caption("Powered by CockroachDB Vector Indexing & AWS")

@st.cache_resource
def load_agent():
    json_matches = list(Path(".").rglob("fulltext_chunks.json"))
    if not json_matches:
        st.error("Corpus data not found! Please check fulltext_chunks.json path.")
        return None
    with open(json_matches[0], "r") as f:
        corpus = json.load(f)
    return RAGAgent(corpus_chunks=corpus)

agent = load_agent()

# Session ID Management for Persistent Memory
if "session_id" not in st.session_state:
    st.session_state.session_id = "hackathon_demo_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Architecture Overview
with st.sidebar:
    st.header("🛠️ Stack & Architecture")
    st.markdown("**Memory Layer:** CockroachDB Cloud (AWS ap-south-1)")
    st.markdown("**Vector Store:** CockroachDB Distributed Vector Indexing")
    st.markdown("**Cloud Engine:** AWS Bedrock / S3 Integration")
    st.markdown("**Control Plane:** CockroachDB ccloud CLI & MCP")
    st.divider()
    st.text_input("Session Key", value=st.session_state.session_id, key="session_id")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if agent:
            with st.spinner("Querying CockroachDB Vector Index & Memory..."):
                res = agent.query(session_id=st.session_state.session_id, question=prompt)
                answer = res["answer"]
                sources = ", ".join(res["sources"])
                full_response = f"{answer}\n\n**Sources:** `{sources}`"
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("Agent failed to initialize.")