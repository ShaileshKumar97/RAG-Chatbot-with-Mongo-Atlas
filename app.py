import os
import time
from dotenv import load_dotenv

from openai import OpenAI
from pymongo import MongoClient

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate

import gradio as gr


# Load environment variables from .env file
load_dotenv(override=True)

# Set up MongoDB connection details
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
MONGODB_NAME = os.environ["MONGODB_NAME"]
MONGODB_COLLECTION = os.environ["MONGODB_COLLECTION"]
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]

# Define field names
EMBEDDING_FIELD_NAME = "embedding"
TEXT_FIELD_NAME = "text"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGODB_NAME]
collection = db[MONGODB_COLLECTION]

def process_pdf(file, progress=gr.Progress()):
    progress(0, desc="Starting")
    # time.sleep(1)
    progress(0.05)
    for letter in progress.tqdm(file.name, desc="Uploading Your PDF into MongoDB Atlas"):
        time.sleep(0.25)

    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()

    # Split text into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)

    # Set up MongoDBAtlasVectorSearch with embeddings
    # Insert the documents into MongoDB Atlas Vector Search
    vectorStore = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, disallowed_special=()),
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    return vectorStore

def query_and_display(query, history):

    history_langchain_format = []

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    history_langchain_format.append(HumanMessage(content=query))

    # Set up MongoDBAtlasVectorSearch with embeddings
    vectorStore = MongoDBAtlasVectorSearch(
        collection,
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    # By default, the vector store retriever uses similarity search. 
    # If the underlying vector store supports maximum marginal relevance search, you can specify that as the search type.
    retriever = vectorStore.as_retriever(
        # search_type="mmr",
        search_type="similarity",
        search_kwargs={"k": 5},
    )
  
    prompt_template = """
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
    )

    # Query MongoDB Atlas Vector Search
    retriever_output = qa.run(query)

    return retriever_output

with gr.Blocks(css=".gradio-container {background-color: AliceBlue}") as demo:
    gr.Markdown("Generative AI Chatbot - Upload your file and Ask questions")

    with gr.Tab("Upload PDF"):
        with gr.Row():
            pdf_input = gr.File()
            pdf_output = gr.Textbox()
        pdf_button = gr.Button("Upload PDF")

    with gr.Tab("Ask question"):
 
        gr.ChatInterface(query_and_display)

    pdf_button.click(process_pdf, inputs=pdf_input, outputs=pdf_output)
    
demo.launch()
