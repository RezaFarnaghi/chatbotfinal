#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import streamlit as st
from document_processing import read_pdf, read_txt, split_doc, embedding_storing
from chatbot import prepare_rag_llm, generate_answer

def load_api_key():
    st.sidebar.header("API Configuration")
    api_key = st.sidebar.text_input("Enter your API Key:", type='password')
    return api_key

def main():
    st.set_page_config(page_title="Enhanced RAG Chatbot", page_icon="🤖", layout="wide")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Select a page:", ["Home", "Document Embedding", "RAG Chatbot"])

    if selection == "Home":
        st.title("Welcome to the Enhanced RAG Chatbot")
        st.write("""
            This project demonstrates the creation of a Retrieval-Augmented Generation (RAG) chatbot using Streamlit.
            Use the sidebar to navigate to different sections.
        """)
    elif selection == "Document Embedding":
        display_document_embedding_page()
    elif selection == "RAG Chatbot":
        display_chatbot_page()

def display_document_embedding_page():
    st.header("Document Embedding")
    st.write("Upload documents to create a custom knowledge base for the chatbot.")
    col1, col2 = st.columns(2)

    with col1:
        document = st.file_uploader("Upload Documents (PDF or TXT)", type=['pdf', 'txt'], accept_multiple_files=True)
        instruct_embeddings = st.text_input("Instruct Embeddings Model", value="sentence-transformers/all-MiniLM-L6-v2")
        chunk_size = st.number_input("Chunk Size", value=200, min_value=0, step=1)
        chunk_overlap = st.number_input("Chunk Overlap", value=10, min_value=0, step=1)

    with col2:
        if not os.path.exists("vector_store"):
            os.makedirs("vector_store")
        vector_store_list = ["<New>"] + [item for item in os.listdir("vector_store") if item != ".DS_Store"]
        existing_vector_store = st.selectbox("Select or Create Vector Store", vector_store_list)
        new_vs_name = st.text_input("New Vector Store Name", value="new_vector_store")
        save_button = st.button("Save Vector Store")

    if save_button:
        if document:
            combined_content = ""
            for file in document:
                if file.name.endswith(".pdf"):
                    combined_content += read_pdf(file)
                elif file.name.endswith(".txt"):
                    combined_content += read_txt(file)

            split = split_doc(combined_content, chunk_size, chunk_overlap)
            create_new_vs = existing_vector_store == "<New>"

            try:
                embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name)
                st.success("Vector store saved successfully!")
            except ValueError as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please upload at least one file.")

def display_chatbot_page():
    st.header("Multi Source Chatbot")

    with st.expander("Initialize the LLM Model"):
        st.write("""
            Configure the chatbot settings. Insert the token and select the vector store, temperature, and maximum character length.
            **NOTE:** Token: API Key from Hugging Face. Temperature: Controls the creativity of the chatbot (Value between 0 and 1)
        """)

        col1, col2 = st.columns(2)
        with col1:
            api_key = load_api_key()
            if api_key:
                token = st.text_input("Hugging Face Token", type='password', value="******")
            else:
                token = st.text_input("Hugging Face Token", type='password')
            llm_model = st.text_input("LLM Model", value="tiiuae/falcon-7b-instruct")

        with col2:
            instruct_embeddings = st.text_input("Instruct Embeddings", value="sentence-transformers/all-MiniLM-L6-v2")
            vector_store_list = os.listdir("vector_store")
            existing_vector_store = st.selectbox("Vector Store", vector_store_list)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            max_length = st.number_input("Maximum Character Length", value=200, min_value=1, step=1)

        create_chatbot = st.button("Initialize Chatbot")

    if create_chatbot:
        if api_key:
            st.session_state.conversation = prepare_rag_llm(api_key, existing_vector_store, temperature, max_length)
            st.success("Chatbot initialized successfully!")
        else:
            st.error("Please enter a valid API key.")

    st.write("### Chat with the Bot")
    st.write("Enter your text below to get a response from the chatbot. **NOTE:** Initialize the LLM Model above before using the chatbot.")

    if 'conversation' in st.session_state:
        user_input = st.text_area("Your Question:", value="", height=150)
        submit_button = st.button("Send")

        if submit_button and user_input:
            with st.spinner("Generating response..."):
                answer, doc_source = generate_answer(user_input, api_key)

            st.write(f"**You:** {user_input}")
            st.write(f"**Chatbot:** {answer}")

            st.write("### Source Documents")
            for idx, source in enumerate(doc_source, 1):
                st.write(f"**{idx}.** {source}")
    else:
        st.warning("Please initialize the chatbot first.")

if __name__ == "__main__":
    main()


# In[ ]:




