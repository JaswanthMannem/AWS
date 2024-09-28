import json
import os
import boto3
import streamlit as st

# Aws titan model for generating embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion( Storing Data )

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embeddings and Vector Stores......

from langchain.vectorstores import FAISS

# Tools for LLM model

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client("bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embeddings and Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")
    
def get_claude_llm():
    llm = Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,model_kwargs={'max_gen_len':512})
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

prompt = PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt}
    )
    answer=qa({"query":query})
    return answer["result"]
    
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁",divider="grey")
    
    user_question = st.text_input("Ask Question from the PDF Files")
    
    with st.sidebar:
        st.title("Update or Create Vector Store:")
        
        if st.button("Vector Update"):
            with st.spinner("Processing.."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")
                
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
            
    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_llama2_llm()
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
            
if __name__=="__main__":
    main()