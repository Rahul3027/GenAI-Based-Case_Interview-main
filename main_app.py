# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv  # Import dotenv to load environment variables
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import uuid

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

app = FastAPI()

# Global variables for simplicity (in production, use database)
vectorstore = None
sessions = {}  # session_id: {"memory": ConversationBufferMemory, "case": str, "messages": list}

# Function to load knowledge base files with error handling
def load_knowledge_base():
    documents = []
    files = [
        "data/zomato_context.txt",
        "data/rca_framework.txt",
        "data/interview_cases.txt"
    ]
    
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        
        try:
            loader = TextLoader(file, autodetect_encoding=True)
            documents.extend(loader.load())
        except Exception as e:
            raise RuntimeError(f"Error loading {file}") from e
    
    return documents

# Define the custom prompt template for interviewer behavior
interviewer_prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template="""
You are an experienced Product Manager at Zomato conducting a Root Cause Analysis interview.
Your job is to actively engage with the candidate's ideas and guide them through structured problem-solving.

Current case: "Zomato has observed that lunch order conversion rate is only 2% compared to 10% for breakfast and dinner."

Your responsibilities:
1. Acknowledge the candidate's ideas with brief feedback.
2. Ask 2-3 specific follow-up questions to probe deeper into their thinking.
3. Guide them through the RCA framework (problem definition, data collection, hypothesis formation, etc.).
4. Challenge their assumptions in a constructive way.
5. Ask for data they would analyze to validate their hypotheses.
6. Push them to explore multiple potential causes, not just one obvious cause.
7. Help them prioritize which causes to address first.

Previous conversation:
{chat_history}

Candidate's latest response: {question}

Relevant knowledge from your context: {context}

Respond as an active interviewer who is testing the candidate's product thinking and analytical skills.
"""
)

# Function to initialize the vector store and embeddings
def initialize_system():
    global vectorstore
    if vectorstore is None:
        documents = load_knowledge_base()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)  # In-memory for Vercel
    
    return vectorstore

# Dictionary of interview cases
case_descriptions = {
    "Lunch Conversion Rate Drop": "Zomato has observed that lunch order conversion rate is only 2% compared to 10% for breakfast and dinner. Why might this be happening and how would you address it?",
    "Delivery Complaints": "There has been a sudden 30% increase in complaints about partial order deliveries. What could be causing this and how would you solve it?",
    "Restaurant Rating Decline": "Average restaurant ratings in Pune have dropped 10% over the last month. How would you investigate and address this issue?",
    "User Retention Problem": "Weekly active users in Bangalore have declined by 15% despite increased marketing spend. What could be the root causes?",
    "Delivery Time Increases": "Average delivery time has increased from 28 to 35 minutes in Delhi NCR region. How would you approach this problem?"
}

def get_session_id(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "case": "Lunch Conversion Rate Drop",
            "messages": [{
                "role": "assistant",
                "content": (
                    f"Welcome to your Zomato Product Manager interview! I'll be evaluating your approach to Root Cause Analysis (RCA) today.\n\n"
                    f"**Case**: {case_descriptions['Lunch Conversion Rate Drop']}\n\n"
                    f"Please begin by:\n"
                    f"1. Defining the problem as you understand it\n"
                    f"2. What initial data would you want to look at?\n"
                    f"3. What are your first hypotheses about potential causes?\n\n"
                    f"Take your time to structure your thoughts before responding."
                )
            }]
        }
    return sessions[session_id]


def render_page(messages, cases):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Zomato PM Interview Simulator</title>
      <style>
        :root {
          color-scheme: dark;
          font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #e2e8f0;
          background: #020617;
        }
        * {
          box-sizing: border-box;
        }
        html, body {
          margin: 0;
          min-height: 100%;
        }
        body {
          background: radial-gradient(circle at top left, rgba(34, 211, 238, 0.16), transparent 28%),
                      radial-gradient(circle at bottom right, rgba(255, 133, 27, 0.16), transparent 34%),
                      linear-gradient(180deg, #060b16 0%, #020617 100%);
          padding: 24px;
          display: flex;
          justify-content: center;
        }
        .container {
          width: 100%;
          max-width: 960px;
          display: flex;
          flex-direction: column;
          gap: 18px;
        }
        .panel {
          background: rgba(15, 23, 42, 0.92);
          border: 1px solid rgba(148, 163, 184, 0.14);
          border-radius: 32px;
          box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
          overflow: hidden;
        }
        header {
          padding: 32px;
          background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.80));
        }
        header h1 {
          margin: 0;
          font-size: clamp(2rem, 2.5vw, 3rem);
          letter-spacing: -0.04em;
          color: #ff8700;
        }
        header p {
          margin: 12px 0 0;
          color: #94a3b8;
          max-width: 720px;
          line-height: 1.75;
        }
        .chat-window {
          padding: 24px;
          display: grid;
          gap: 14px;
        }
        .message {
          padding: 18px 20px;
          border-radius: 22px;
          line-height: 1.75;
          white-space: pre-wrap;
          word-break: break-word;
          max-width: 95%;
        }
        .assistant {
          background: rgba(30, 41, 59, 0.96);
          border: 1px solid rgba(56, 189, 248, 0.16);
          color: #f8fafc;
        }
        .user {
          justify-self: end;
          background: linear-gradient(135deg, #2563eb, #1e40af);
          color: #ffffff;
          border: 1px solid rgba(96, 165, 250, 0.24);
        }
        .input-panel {
          padding: 24px;
          display: grid;
          gap: 16px;
          border-top: 1px solid rgba(148, 163, 184, 0.10);
        }
        .input-panel form {
          display: grid;
          gap: 14px;
        }
        input[type="text"], select {
          width: 100%;
          padding: 16px 18px;
          border-radius: 16px;
          border: 1px solid rgba(148, 163, 184, 0.18);
          background: rgba(15, 23, 42, 0.88);
          color: #e2e8f0;
          font-size: 1rem;
          outline: none;
        }
        input[type="text"]:focus,
        select:focus {
          border-color: #38bdf8;
          box-shadow: 0 0 0 5px rgba(56, 189, 248, 0.12);
        }
        .controls {
          display: grid;
          grid-template-columns: 1fr auto;
          gap: 14px;
          align-items: stretch;
        }
        button {
          min-height: 54px;
          border: none;
          border-radius: 16px;
          background: linear-gradient(135deg, #fb923c, #f97316);
          color: white;
          font-weight: 700;
          cursor: pointer;
          transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        button:hover {
          transform: translateY(-1px);
          box-shadow: 0 20px 40px rgba(251, 146, 60, 0.20);
        }
        @media (max-width: 720px) {
          body { padding: 16px; }
          .container { gap: 14px; }
          .controls { grid-template-columns: 1fr; }
          button { width: 100%; }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="panel">
          <header>
            <h1>🍱 Zomato PM Interview Simulator</h1>
            <p>Practice Product Manager interview cases and Root Cause Analysis using AI-guided conversation.</p>
          </header>
          <div class="chat-window">
    """
    for msg in messages:
        html_content += f'<div class="message {msg["role"]}">{msg["content"]}</div>'
    html_content += """
          </div>
          <div class="input-panel">
            <form action="/send" method="post">
              <input type="text" name="message" placeholder="Enter your response..." required />
              <button type="submit">Send Message</button>
            </form>
            <form action="/restart" method="post">
              <div class="controls">
                <select name="case">
    """
    for case in cases:
        html_content += f'<option value="{case}">{case}</option>'
    html_content += """
                </select>
                <button type="submit">Restart Interview</button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    return html_content

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = get_session_id(request)
    session = get_session(session_id)
    response = HTMLResponse(content=render_page(session["messages"], case_descriptions.keys()))
    response.set_cookie("session_id", session_id)
    return response

@app.post("/send")
async def send_message(request: Request, message: str = Form(...)):
    session_id = get_session_id(request)
    session = get_session(session_id)
    
    vectorstore = initialize_system()
    llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.7)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=session["memory"],
        combine_docs_chain_kwargs={"prompt": interviewer_prompt},
        get_chat_history=lambda h: h,
        verbose=True,
    )
    
    session["messages"].append({"role": "user", "content": message})
    
    try:
        response = chain.invoke({"question": message})
        assistant_message = response["answer"]
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "resource_exhausted" in error_str:
            assistant_message = (
                "I apologize, but I've reached my API quota limit. As a simulated interviewer, let me provide some guidance based on common RCA practices:\n\n"
                "For the lunch conversion rate issue, here are some potential root causes to consider:\n"
                "1. **Menu availability**: Restaurants might have limited lunch menus or higher prices\n"
                "2. **Delivery capacity**: Fewer delivery partners available during lunch hours\n"
                "3. **User behavior**: People might prefer eating out or have less time for online ordering\n\n"
                "What data would you look at to validate these hypotheses?"
            )
        else:
            assistant_message = (
                "I apologize for the technical difficulty. Please try again or rephrase your response. "
                "If the issue persists, please check that your API key is valid and has available quota."
            )
    
    session["messages"].append({"role": "assistant", "content": assistant_message})
    response = HTMLResponse(content=render_page(session["messages"], case_descriptions.keys()))
    response.set_cookie("session_id", session_id)
    return response

@app.post("/restart")
async def restart_interview(request: Request, case: str = Form(...)):
    session_id = get_session_id(request)
    sessions[session_id] = {
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        "case": case,
        "messages": [{
            "role": "assistant",
            "content": (
                f"Welcome to your Zomato Product Manager interview! I'll be evaluating your approach to Root Cause Analysis (RCA) today.\n\n"
                f"**Case**: {case_descriptions[case]}\n\n"
                f"Please begin by:\n"
                f"1. Defining the problem as you understand it\n"
                f"2. What initial data would you want to look at?\n"
                f"3. What are your first hypotheses about potential causes?\n\n"
                f"Take your time to structure your thoughts before responding."
            )
        }]
    }
    html_content = f"""
    <html>
    <head><title>Zomato PM Interview Simulator</title></head>
    <body>
    <h1>Zomato PM Interview Simulator</h1>
    <div id="chat-messages">
    """
    for msg in sessions[session_id]["messages"]:
        html_content += f'<div class="message {msg["role"]}">{msg["content"]}</div>'
    html_content += """
    </div>
    <form action="/send" method="post">
        <input type="text" name="message" placeholder="Your response" required>
        <button type="submit">Send</button>
    </form>
    <form action="/restart" method="post">
        <select name="case">
    """
    for case in case_descriptions.keys():
        html_content += f'<option value="{case}">{case}</option>'
    html_content += """
        </select>
        <button type="submit">Restart Interview</button>
    </form>
    </body>
    </html>
    """
    response = HTMLResponse(content=render_page(sessions[session_id]["messages"], case_descriptions.keys()))
    response.set_cookie("session_id", session_id)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
