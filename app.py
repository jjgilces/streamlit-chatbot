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
from langchain.document_loaders import TextLoader, CSVLoader

# Set OPENAI_API_KEY as an environment variable
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0.1)

def load_csv_data(path):
    try:
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
    except Exception as e:
        st.error(f"Error al cargar datos CSV: {e}")
        return []

def load_txt_data(path):
    try:
        text_loader = TextLoader(path)
        data = text_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len
        )
        return text_splitter.transform_documents(data)
    except Exception as e:
        st.error(f"Error al cargar datos de texto: {e}")
        return []

def create_embeddings_and_vector_store(directory_documents):
    try:
        store = LocalFileStore("./cache/")
        embed_model_id = 'paraphrase-multilingual-MiniLM-L12-v2'

        core_embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id)
        embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model, store, namespace=embed_model_id)

        return FAISS.from_documents(directory_documents, embedder)
    except Exception as e:
        st.error(f"Error al crear embeddings y almacenamiento de vectores: {e}")
        return None

def create_retrieval_chain(llm, vector_store):
    try:
        handler = StdOutCallbackHandler()
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            callbacks=[handler],
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error al crear la cadena de recuperación: {e}")
        return None

def conversation_chat(query, chain, history):
    try:
        output = chain({"query": query})
        history.append((query, output['result']))
        return output['result']
    except Exception as e:
        st.error(f"Error en la conversación del chat: {e}")
        return "Lo siento, ocurrió un error."

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Pregunta:", placeholder="Pregúntame lo que quieras", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            with st.spinner('Obteniendo respuesta...'):
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
        st.session_state['generated'] = ["Hola! ¿Qué te gustaría saber?"]
        st.session_state['past'] = ["Hola! 👋"]

    st.title("Chatbot FIEC :books:")
    directory_documents = load_csv_data('data/directorio.csv')
    base_documents = load_txt_data('data/base.txt') 
    teachers_documents= load_txt_data('data/coordinadores.txt')
    print(teachers_documents)
    if not directory_documents or not base_documents:
        st.error("Error al cargar los documentos. Por favor verifica los archivos y formatos.")
        return

    combined_documents =   base_documents+ teachers_documents + directory_documents
    vector_store = create_embeddings_and_vector_store(combined_documents)
    if not vector_store:
        st.error("Error al crear el almacenamiento de vectores.")
        return

    chain = create_retrieval_chain(llm, vector_store)
    if not chain:
        st.error("Error al inicializar la cadena de recuperación.")
        return

    display_chat_history(chain)

if __name__ == "__main__":
    main()
