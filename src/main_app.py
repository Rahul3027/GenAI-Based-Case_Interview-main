# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from dotenv import load_dotenv  # Import dotenv to load environment variables
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in the .env file.")
    raise ValueError("Google API Key not found.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

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
            st.error(f"Missing required file: {file}")
            raise FileNotFoundError(f"File not found: {file}")
        
        try:
            loader = TextLoader(file, autodetect_encoding=True)
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
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
    documents = load_knowledge_base()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")  # Added persistence directory
    
    return vectorstore

# Dictionary of interview cases
case_descriptions = {
    "Lunch Conversion Rate Drop": "Zomato has observed that lunch order conversion rate is only 2% compared to 10% for breakfast and dinner. Why might this be happening and how would you address it?",
    "Delivery Complaints": "There has been a sudden 30% increase in complaints about partial order deliveries. What could be causing this and how would you solve it?",
    "Restaurant Rating Decline": "Average restaurant ratings in Pune have dropped 10% over the last month. How would you investigate and address this issue?",
    "User Retention Problem": "Weekly active users in Bangalore have declined by 15% despite increased marketing spend. What could be the root causes?",
    "Delivery Time Increases": "Average delivery time has increased from 28 to 35 minutes in Delhi NCR region. How would you approach this problem?"
}

# Main function for the Streamlit app
def main():
    # Set up page configuration with custom theme
    st.set_page_config(
        page_title="Zomato PM Interview Simulator",
        page_icon="🍱",  # Restored original emoji icon here
        layout="wide"
    )
    
    # Header with Zomato branding (with emoji)
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>🍱 Zomato PM Interview Simulator</h1>", unsafe_allow_html=True)

    # Custom CSS for styling
    st.markdown("""
        <style>
            body {
                background-color: #121212;
                color: white;
            }
            .stChatMessage {
                border-radius: 15px;
                padding: 10px;
                margin-bottom: 10px;
                font-family: Arial, sans-serif;
            }
            .stChatMessage.user {
                background-color: #1E1E1E;
                color: white;
                border-left: 5px solid #FF5733;
            }
            .stChatMessage.assistant {
                background-color: #F5F5F5;
                color: black;
                border-left: 5px solid #FFC300;
            }
            footer {
                visibility: hidden;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize the system (load knowledge base and create vector store)
    try:
        vectorstore = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize the system: {str(e)}")
        st.error("Please check the knowledge base files.")
        return
    
    # Set up the language model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    
    # Create memory to store chat history
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create conversational chain with custom interviewer prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": interviewer_prompt},
        get_chat_history=lambda h: h,
        verbose=True,
    )
    
    # Sidebar for case selection and controls
    with st.sidebar:
        st.header("Interview Settings")
        
        cases = list(case_descriptions.keys())
        
        selected_case = st.selectbox("Choose a case:", cases, index=0)
        
        if st.button("Restart Interview"):
            st.session_state.memory.clear()
            st.session_state.messages = [{
                "role": "assistant",
                "content": (
                    f"Welcome to your Zomato Product Manager interview! I'll be evaluating your approach to Root Cause Analysis (RCA) today.\n\n"
                    f"**Case**: {case_descriptions[selected_case]}\n\n"
                    f"Please begin by:\n"
                    f"1. Defining the problem as you understand it\n"
                    f"2. What initial data would you want to look at?\n"
                    f"3. What are your first hypotheses about potential causes?\n\n"
                    f"Take your time to structure your thoughts before responding."
                )
            }]
            st.rerun()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Welcome to your Zomato Product Manager interview! I'll be evaluating your approach to Root Cause Analysis (RCA) today.\n\n"
                f"**Case**: {case_descriptions[selected_case]}\n\n"
                f"Please begin by:\n"
                f"1. Defining the problem as you understand it\n"
                f"2. What initial data would you want to look at?\n"
                f"3. What are your first hypotheses about potential causes?\n\n"
                f"Take your time to structure your thoughts before responding."
            )
        }]

    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role_class):
            st.write(msg["content"])

    if prompt := st.chat_input("Your response"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            response = chain.invoke({"question": prompt})
            assistant_message = response["answer"]
        except Exception as e:
            assistant_message = (
                f"I apologize for the technical difficulty. Please try again."
            )
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})

if __name__ == "__main__":
    main()
