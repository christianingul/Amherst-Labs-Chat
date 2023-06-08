from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from typing import Any, List
import pinecone
import streamlit as st



#[]: This empty square bracket notation is used to indicate that the variable List is initialized as an empty list.
def run_llm(query: str, chat_history: List[tuple[str, Any]] = []) -> Any:

    pinecone_api_key = st.secrets.get("PINECONE")
    pinecone_environment = st.secrets.get("ENVIRONMENT")
    index_name = st.secrets.get("INDEX")
    openai_api_key = st.secrets.get("OPENAI")



    pinecone.init(
        api_key = pinecone_api_key,
        environment = pinecone_environment
    )



    if index_name not in pinecone.list_indexes():

        loader = TextLoader(file_path="/Users/christianbjorkingul/PycharmProjects/pythonProject23/knowledge_base.txt")
        raw_text = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap  = 40,
            length_function= len
        )

        chunks = text_splitter.split_documents(raw_text)
        print(f"Splitted into {len(chunks)} chunks")

        embeddings = OpenAIEmbeddings(model= "text-embedding-ada-002", openai_api_key=openai_api_key)

        knowledge_base = Pinecone.from_documents(embedding=embeddings, documents=chunks, index_name=index_name)

        print(f"Added {len(chunks)} chunks to Pinecone vector store '{index_name}'")

    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",
                                      openai_api_key=openai_api_key )
        openai_api_key=openai_api_key

        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    chat = ChatOpenAI(verbose=True, temperature=0, openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo')

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})











