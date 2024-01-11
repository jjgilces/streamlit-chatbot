import streamlit as st
from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import StdOutCallbackHandler
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader,CSVLoader


# Set OPENAI_API_KEY as an environment variable
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

def load_csv_data(path):
    csv_loader = CSVLoader(file_path=path, csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['Oficina', 'Nombre', 'Descripcion', 'Telefono']
    })
    directory_data = csv_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.transform_documents(directory_data)




def load_txt_data(path):
    text_loader = TextLoader(path)
    data = text_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len
    )

    return text_splitter.transform_documents(data)

def create_embeddings_and_vector_store(directory_documents):
    store = LocalFileStore("./cache/")
    embed_model_id = 'paraphrase-multilingual-MiniLM-L12-v2'

    core_embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id)
    embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model, store, namespace=embed_model_id)

    return FAISS.from_documents(directory_documents, embedder)

def create_retrieval_chain(llm, vector_store):
    handler = StdOutCallbackHandler()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        callbacks=[handler],
        return_source_documents=True
    )

def conversation_chat(query, chain, history):
    output = chain({"query": query})
    history.append((query, output['result']))
    return output['result']

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me anything", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Loading answer...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])


            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def main():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        st.session_state['generated'] = ["Hello! Ask me anything"]
        st.session_state['past'] = ["Hey! 👋"]

    st.title("Chatbot FIEC :books:")
    
    # Load CSV data
    directory_documents = load_csv_data('data/directorio.csv')

    txt_documents = load_txt_data('data/base.txt')
    combined_documents = directory_documents   + txt_documents 

    # Create embeddings and vector store
    vector_store = create_embeddings_and_vector_store(combined_documents)

    # Create the chain object
    chain = create_retrieval_chain(llm, vector_store)

    # Display chat history
    display_chat_history(chain)

if __name__ == "__main__":
    main()
