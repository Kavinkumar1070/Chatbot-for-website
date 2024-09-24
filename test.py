import os
import bs4
import streamlit as st

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG_With_Memory"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCXx_qxtPPghstnxkm8ehgU_N_hhmjvmq0"
os.environ["GROQ_API_KEY"] = "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"


def main():
    st.title('🦜🔗 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    prompt = st.text_input("Ask a question about services provided by Converse Data Solutions")
    if st.button("Submit Query", type="primary"):
        ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        DB_DIR = os.path.join(ABS_PATH, "db")

        # Load website content
        loader = WebBaseLoader(web_paths=("https://www.conversedatasolutions.com/",))
        doc = loader.load()

        if len(doc) == 0:
            st.error("No documents were retrieved from the website.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(doc)

        if len(splits) == 0:
            st.error("No document chunks were created.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=DB_DIR
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Log the user's prompt
        st.write("User Query:", prompt)

        # Get relevant documents based on the user's query
        relevant_docs = retriever.get_relevant_documents(prompt)

        if len(relevant_docs) == 0:
            st.error("No relevant documents were retrieved from the vectorstore.")
            return

        # Display retrieved documents for debugging
        for i, doc in enumerate(relevant_docs):
            st.write(f"Retrieved Document {i+1}:", doc.page_content)

        #model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", convert_system_message_to_human=True)
        model = ChatGroq(
                    model="mixtral-8x7b-32768",
                    temperature=0,
                    max_tokens=None,
                    timeout=None)
        # Refine the system prompt to specifically focus on answering the question
        system_prompt = (
            "You are an assistant. Answer the question based on the following context. "
            "If the answer is not available in the context, say you don't know.\n\n{context}"
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

        # Creating the chain
        question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answering_chain)

        # Process the input query and display the result
        context = "\n".join(doc.page_content for doc in relevant_docs)
        response = rag_chain.invoke({"input": prompt, "context": context})

        result = response.get("answer", "No answer found.")
        st.write("Response:", result)

if __name__ == '__main__':
    main()
