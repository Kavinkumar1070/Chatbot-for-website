import os
import bs4
from fastapi import FastAPI, HTTPException,Request
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates


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

# Create FastAPI app
app = FastAPI()

# Serve static files from the frontend directory
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def home(request: Request):
    # Serve the base.html directly from the frontend folder
    return FileResponse("frontend/base.html")

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for input request
class QueryRequest(BaseModel):
    question: str

# Predefined questions and their responses
predefined_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello! How can I assist you today?",
    "hey": "Hi there! How can I help you?",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Take care!",
    "help": "I'm here to help! You can ask me about our company.",
    "thanks": "You're welcome! If you have more questions, feel free to ask.",
    "thank you": "You're welcome! I'm here to assist you.",
    "how are you?": "I'm just a program, but thanks for asking! How can I assist you?",
    "what's up?": "Not much! How can I help you today?",
    "what can you do?": "I can assist you with questions about our company and services.",
    "tell me about yourself": "I'm a virtual assistant here to help with your queries.",
    "can you help me?": "Of course! Please ask your question.",
    "hi there": "Hello! How can I help you today?",
    "good morning": "Good morning! How can I assist you?",
    "good evening": "Good evening! How can I help you?",
    "welcome": "Welcome! How can I assist you today?"
}


# Load the website content and prepare the database
def prepare_database():
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    # # Load website content
    # loader = WebBaseLoader(web_paths=("https://www.conversedatasolutions.com/",))
    # doc = loader.load()
    
    important_pages = [
    "https://www.conversedatasolutions.com/technology",
    "https://www.conversedatasolutions.com/contact-us",
    "https://www.conversedatasolutions.com/services/data-visualization",
    "https://www.conversedatasolutions.com/services/web-development",
    "https://www.conversedatasolutions.com/services/data-engineering",
    "https://www.conversedatasolutions.com/services/digital-lcnc",
    "https://www.conversedatasolutions.com/services/data-science&machine-learning",
    "https://www.conversedatasolutions.com/services/artificial-intelligence"
    ]
    loader = WebBaseLoader(web_paths=important_pages)
    important_docs = loader.load()
    
    loader = WebBaseLoader(web_paths=("https://www.conversedatasolutions.com/",))
    full_site_docs = loader.load()
    
    all_docs = important_docs + full_site_docs

    if len(all_docs) == 0:
        raise Exception("No documents were retrieved from the website.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    if len(splits) == 0:
        raise Exception("No document chunks were created.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    return vectorstore

# Prepare the vectorstore at startup
vectorstore = prepare_database()


@app.post("/ask")
async def ask_question(query: QueryRequest):
    prompt = query.question.lower()  # Convert question to lowercase to handle case insensitivity

    # Check if the question matches any predefined responses
    if prompt in predefined_responses:
        return {"response": predefined_responses[prompt]}
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Get relevant documents based on the user's query
    relevant_docs = retriever.get_relevant_documents(prompt)

    if len(relevant_docs) == 0:
        raise HTTPException(status_code=404, detail="No relevant documents were retrieved from the vectorstore.")

    # Display retrieved documents for debugging
    context = "\n".join(doc.page_content for doc in relevant_docs)

    #model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", convert_system_message_to_human=True)
    model = ChatGroq(
                    model="mixtral-8x7b-32768",
                    temperature=0,
                    max_tokens=None,
                    timeout=None)
    # Refine the system prompt to specifically focus on answering the question
    system_prompt = (
    "You are a helpful assistant for Converse Data Solutions. Your responses depend on the user's query:\n"
    "1. If the query relates to website details, provide a concise answer based on the context.\n"
    "2. If the query is unclear or doesn't have meaning, respond with a concise request for clarification.\n"
    "3. If you don't know the answer, simply say you don't know.\n"
    "Use clear and concise language based on the following context:\n\n{context}\n\n"
)



    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # Creating the chain
    question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)

    # Process the input query and display the result
    response = rag_chain.invoke({"input": prompt, "context": context})

    result = response.get("answer", "No answer found.")
    print(result)
    return {"response": result}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
