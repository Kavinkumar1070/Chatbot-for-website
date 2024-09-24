import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file (Optional)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# FastAPI app
app = FastAPI()

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

# Create prompt templates
class QueryRequest(BaseModel):
    prompt: str

@app.post("/process_website/")
async def process_website(query: QueryRequest):
    try:
        # Extract inputs from the request body
        url = "https://www.conversedatasolutions.com/"
        prompt = query.prompt

        # Create directory for db
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=500,
                                              chunk_overlap=40)
        docs = text_splitter.split_documents(data)

        # Create HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create Chroma vector database
        vectordb = Chroma.from_documents(documents=docs,
                                         embedding=embeddings,
                                         persist_directory=DB_DIR)

        vectordb.persist()

        # Create retriever from Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use ChatGroq model
        llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2)

        # Create RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Run the prompt and return the response
        response = qa(prompt)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using Uvicorn (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
