import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {bool(GOOGLE_API_KEY)}")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def load_knowledge_base():
    documents = []
    files = [
        "data/zomato_context.txt",
        "data/rca_framework.txt",
        "data/interview_cases.txt"
    ]
    
    for file in files:
        if not os.path.exists(file):
            print(f"Missing file: {file}")
            return None
        try:
            loader = TextLoader(file, autodetect_encoding=True)
            documents.extend(loader.load())
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return None
    return documents

def initialize_system():
    documents = load_knowledge_base()
    if not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        print("Vectorstore created successfully")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None

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

def test_chain():
    vectorstore = initialize_system()
    if not vectorstore:
        return
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.7)
        print("LLM initialized")
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": interviewer_prompt},
            get_chat_history=lambda h: h,
            verbose=True,
        )
        print("Chain created successfully")
        
        # Test a simple query
        response = chain.invoke({"question": "Hello"})
        print(f"Response: {response['answer']}")
        
    except Exception as e:
        print(f"Error in chain: {e}")

if __name__ == "__main__":
    test_chain()