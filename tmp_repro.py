import os
from dotenv import load_dotenv
load_dotenv()
print('GOOGLE_API_KEY', os.getenv('GOOGLE_API_KEY') is not None)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

files = ['data/zomato_context.txt', 'data/rca_framework.txt', 'data/interview_cases.txt']
docs = []
for f in files:
    print('loading', f)
    if not os.path.exists(f):
        raise FileNotFoundError(f'Missing {f}')
    docs.extend(TextLoader(f, autodetect_encoding=True).load())
print('docs', len(docs))
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
print('chunks', len(chunks))
emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print('starting chroma')
vect = Chroma.from_documents(chunks, emb, persist_directory='./chroma_db')
print('vectorstore ready')
prompt = PromptTemplate(input_variables=['question', 'chat_history', 'context'], template='Question: {question}\nHistory: {chat_history}\nContext: {context}')
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.2)
mem = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vect.as_retriever(search_kwargs={'k': 3}),
    memory=mem,
    combine_docs_chain_kwargs={'prompt': prompt},
    get_chat_history=lambda h: h,
    verbose=True,
)
q = 'Why is lunch conversion low?'
print('asking')
res = chain.invoke({'question': q})
print('res', res)
print('answer', res.get('answer'))
