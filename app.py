import json
import os
import sys
import boto3
import streamlit as st


from langchain_community.embeddings import BedrockEmbeddings

from langchain.llms.bedrock import Bedrock 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from QASystem.data_ingestion import data_ingestion
from QASystem.data_ingestion import get_vector_store
from QASystem.retrivaland_generation import get_llama2_llm,get_response_llm


region_name = "us-east-1"
bedrock = boto3.client(service_name = "bedrock-runtime",  region_name=region_name)
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

def main():
    st.set_page_config("QA with docs")
    st.header("QA with docs using langchain")

    user_question = st.text_input("ask a question")

    with st.sidebar:
        st.title("update or create the vector store")
        if st.button("vector updated"):
            with st.spinner("processing"):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("vector store updated successfully")
        
        if st.button("llama model"):
            with st.spinner("processing"):
                faiss_index = FAISS.load_local("./faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm=get_llama2_llm()
                
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")

if __name__=="__main__":
    main()
    