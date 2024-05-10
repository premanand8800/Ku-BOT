# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb
import getpass
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass()

groq_api_key = os.environ['GROQ_API_KEY']




def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks,embeddings )

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name='mixtral-8x7b-32768')
   # llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 

    llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name='mixtral-8x7b-32768'
    )
    
   # llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="KU-BOT", page_icon="🤖")
st.title("KU-BOT 🤖")

# Predefined list of URLs


url_list=[
    "https://elec.ku.edu.np/people",
    "https://elec.ku.edu.np/laboratories",
    "https://elec.ku.edu.np/programs",
    "https://elec.ku.edu.np/research",
    "https://elec.ku.edu.np/about-us",
    "https://elec.ku.edu.np/message-from-hod",
    "https://soe.ku.edu.np/about-us",
    "https://soe.ku.edu.np/message-from-the-dean",
    "https://soe.ku.edu.np/collaborative-institutes",
    "https://soe.ku.edu.np/qaa",
    "https://soe.ku.edu.np/faculty-board",
    "https://soe.ku.edu.np/research-committee",
    "https://civil.ku.edu.np/about-dce",
    "https://civil.ku.edu.np/message-from-the-hod",
    "https://civil.ku.edu.np/be-in-civil-engineering-1091",
    "https://civil.ku.edu.np/bachelor-of-architecture",
    "https://civil.ku.edu.np/be-in-mining-engineering",
    "http://kusoa.edu.np",
  "https://www.youtube.com/user/KathmanduUniversity",
  "https://som.ku.edu.np/",
  "https://kusms.edu.np",
  "https://ku.edu.np/finance",
  "https://ku.edu.np/fee-structure",
  "https://ku.edu.np/scholarships-aid",
  "https://ku.edu.np/publications",
  "http://exam.ku.edu.np/",
  "http://rdi.ku.edu.np",
  "https://ku.edu.np/contact/school?site_name=kuhome",
  "http://soe.ku.edu.np",
  "http://sos.ku.edu.np",
  "https://ku.edu.np/collaborative-programs",
  "https://ku.edu.np/endowment-fund",
  "https://ku.edu.np/students",
  "http://alumni.ku.edu.np/",
  "https://ku.edu.np/student-activities",
  "https://ku.edu.np/library",
  "https://ku.edu.np/information",
  "https://ku.edu.np/canteen",
  "https://ku.edu.np/student-hostel",
  "https://ku.edu.np/student-clubs",
  "https://ku.edu.np/academic-calender",
  "https://elibrary.ku.edu.np",
  "https://ku.edu.np/rms",
  "https://ku.edu.np/research-committees",
  "https://ku.edu.np/downloads",
  "https://ku.edu.np/features/media-studies-at-ku",
  "https://kusoed.edu.np/journal/index.php/je",
  "https://ku.edu.np/student-hostels",
  "https://ku.edu.np/student-welfare-council-swc",
  "https://www.facebook.com/kathmanduniversity",
  "https://ku.edu.np/news-app?search_category=4&search_school=10&search_site_name=kuhome",
  "https://crrp.ku.edu.np",
  "https://ku.edu.np/policy-guidelines",
  "https://ku.edu.np/student-clubs-1206",
  "https://ku.edu.np/senate",
  "https://ku.edu.np/student-survey-reports",
  "https://community.ku.edu.np/collaboration/",
  "https://ku.edu.np/student-welfare",
  "https://ku.edu.np/academic-council",
  "https://exam.ku.edu.np/?page_id=4811",
  "https://ku.edu.np/ku-mail",
  "http://journals.ku.edu.np/kuset",
  "https://ku.edu.np/reports",
  "https://ku.edu.np/insider",
  "https://ku.edu.np/canteen-1203",
  "https://ku.edu.np/admission",
  "https://ku.edu.np/academic-programs"
    


]

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a KU-bot. How can I help you?")
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(url_list[0])

# User input
user_query = st.text_input("Type your message here..." ,key="user_input")

if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
