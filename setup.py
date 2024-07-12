from setuptools import find_packages, setup

setup(
    name="qasystem",
    version="0.0.1",
    author="cyril",
    author_email="cyriljopsecky@gmail.com",
    packages=find_packages(),
    install_requires=["langchain", "langchainhub","bs4","tiktoken","openai","boto3","langchain_community","chromadb","awscli","streamlit","pypdf","faiss-cpu"]
)