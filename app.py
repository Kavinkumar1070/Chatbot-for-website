import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
#from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
#from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
#from langchain.chains import create_retrieval_chain

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# Load environment variables from .env file (Optional)
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def main():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    #url = st.text_input("Insert The website URL")

    prompt = st.text_input("Ask a question about Conversedatasolutions website")
    if st.button("Submit Query", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader("https://www.conversedatasolutions.com/")
        data = loader.load()

        # Split the loaded data
        text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=500, 
                                        chunk_overlap=40)

        docs = text_splitter.split_documents(data)

        # Create OpenAI embeddings
        #openai_embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Create a Chroma vector database from the documents
        vectordb = Chroma.from_documents(documents=docs, 
                                        embedding=embeddings,
                                        persist_directory=DB_DIR)

        vectordb.persist()

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatOpenAI model
        llm = ChatGroq(
                    model="mixtral-8x7b-32768",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2)

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


        # Run the prompt and return the response
        response = qa(prompt)
        st.write(response)
        

if __name__ == '__main__':
    main()