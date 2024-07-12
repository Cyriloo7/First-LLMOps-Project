from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock 
import boto3
import json
import sys
from langchain.prompts import PromptTemplate
from QASystem.data_ingestion import get_vector_store
from QASystem.data_ingestion import data_ingestion

region_name = "us-east-1"
bedrock = boto3.client(service_name = "bedrock-runtime",  region_name=region_name)
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT=PromptTemplate( 
    template=prompt_template,input_variables=["context","question"]
)



def get_llama2_llm():
    llm = Bedrock(model_id ="meta.llama2-13b-chat-v1", client=bedrock)
    return llm

def get_response_llm(llm, vectorstores_faiss, query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstores_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
        
    )

    answer=qa({"query":query})
    return answer["result"]


if __name__ =="__main__":
    #docs = data_ingestion()
    #vectorstores_faiss = get_vector_store(docs)
    faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings, allow_dangerous_deserialization=True)
    query = "what is rag tocken"
    llm = get_llama2_llm()
    get_response_llm(llm,faiss_index,query)