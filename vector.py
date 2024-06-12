import os
import openai
import sys
import shutil
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np
from langchain_community.vectorstores import Chroma

sys.path.append('../..')
persist_directory = 'docs/chroma/'

def clear_directory(directory):
    try:
        shutil.rmtree(directory)
        print("Directory removed successfully.")
    except FileNotFoundError:
        print("Directory does not exist.")
    except Exception as e:
        print(f"Error: {e}")

def load_docs():
    loaders = [
    PyPDFLoader("docs/Ant Group (A) 122003-PDF-ENG.pdf"),
    PyPDFLoader("docs/Digital Innovation Slides.pdf"),
    PyPDFLoader("docs/Release Notes for Version 1.20 R1.pdf"),
    PyPDFLoader("docs/ReUp Slides.pdf"),
    PyPDFLoader("docs/Telepass Slides.pdf"),
    PyPDFLoader("docs/Telepass- From Tolling to Mobility Platform (Abridged) 622050-PDF-ENG.pdf")
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

# Split
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 10,
    separators=["\n\n", "\n", "\. ", " ", ""]
     )

    splits = text_splitter.split_documents(docs)
    print("Splits length: ", len(splits))
    return splits

# Example embedding
def sample_embedding(embedding):
    sentence1 = "i like dogs"
    sentence2 = "i like canines"
    sentence3 = "the weather is ugly outside"

    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)


    print(np.dot(embedding1, embedding2))
    print(np.dot(embedding1, embedding3))
    print(np.dot(embedding2, embedding3))
    print(np.dot(embedding1, embedding1))

def create_vector_store(persist_directory, splits, embedding):
    vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
    )

    print("Vector DB collection count: ", vectordb._collection.count())
    return vectordb


# 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
openai.api_key  = os.environ['OPENAI_API_KEY']
print(openai.api_key)

# Load PDF documents
docs = load_docs()
splits = split_documents(docs)

embedding = OpenAIEmbeddings()

sample_embedding(embedding)
clear_directory(persist_directory)

vectordb = create_vector_store(persist_directory, splits, embedding)

question = "vulnerabilities"
docs = vectordb.similarity_search(question,k=3)

print("Question: ", question, "\n")
print("Length of answers: ", len(docs))
i = 1
for doc in docs:
    print("Answer", i, ": ", doc.page_content, "\n")
    i=i+1
# vectordb.persist() #  automatically persisted


