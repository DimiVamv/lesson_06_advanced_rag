"""
app.py — Το AI app του μαθήματος 5, έτοιμο για deployment.

Στο μάθημα 6 δεν τροποποιούμε τον κώδικα του app. Αυτό που κάνουμε:
  - το ανεβάζουμε στο Streamlit Community Cloud
  - χρησιμοποιούμε st.secrets για το ANTHROPIC_API_KEY
  - στο Μέρος Β' βελτιώνουμε το RAG και μετράμε τη βελτίωση (offline)
"""

import streamlit as st
import rag_core

st.title("Ρώτα το έγγραφό σου 💬")

# --- Sidebar: ανέβασμα εγγράφου -------------------------------------------
with st.sidebar:
    st.subheader("📄 Έγγραφο")
    file = st.file_uploader("Ανέβασε PDF ή TXT", type=["pdf", "txt"])
    if st.button("Διάβασέ το") and file:
        chunks = rag_core.load_and_chunk_upload(file.getvalue(), file.name)
        rag_core.index_chunks(chunks, file.name)
        st.success(f"Έτοιμο! {len(chunks)} chunks")

# --- session_state: ιστορικό συζήτησης (προσωρινό, χάνεται σε refresh) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay όλου του ιστορικού σε κάθε rerun του Streamlit
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Νέα ερώτηση ----------------------------------------------------------
question = st.chat_input("Ρώτησέ με κάτι...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    chunks = rag_core.retrieve(question, k=3)
    with st.chat_message("assistant"):
        answer = st.write_stream(rag_core.stream_answer(question, chunks))

    st.session_state.messages.append({"role": "assistant", "content": answer})
