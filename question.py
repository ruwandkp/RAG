import os
import openai
import sys
sys.path.append('../..')
import datetime

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from colorPrint import printc, Color
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


openai.api_key  = os.environ['OPENAI_API_KEY']

llm_name = "gpt-3.5-turbo"
print(llm_name)

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())

llm = ChatOpenAI(model_name=llm_name, temperature=0)

def retrieval_qa(llm_name, vectordb, question):
    docs = vectordb.similarity_search(question,k=10)
    printc(Color.YELLOW, f"Documents length :{len(docs)}")  
  
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    result = qa_chain.invoke({"query": question})

    printc(Color.YELLOW, f"Question: {question}")
    printc(Color.GREEN, f"Answer: {result["result"]}")
    printc(Color.MAGENTA, f"\n{'-' * 100}\n")

from langchain.prompts import PromptTemplate

def prompt_qa(vectordb, llm, question):
    # Build prompt
    # Use ten sentences maximum.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context} 
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain.invoke({"query": question})
    printc(Color.YELLOW, question)
    printc(Color.BLUE, result["result"])
    printc(Color.GREEN, result["source_documents"][0])

question = "What are the known issues in release 1.20?"
prompt_qa(vectordb, llm, question)

question = "What are the special notes included in release 1.20?"
prompt_qa(vectordb, llm, question)

question = "How Ant process user transactions quickly and efficiently?"
prompt_qa(vectordb, llm, question)

question = "What is the share percentage of Ant in MyBank?"
prompt_qa(vectordb, llm, question)

question = "When Ant expanded into insurance?"
prompt_qa(vectordb, llm, question)

# question = "What is the pioneered concept of Sephora?"
# retrieval_qa(llm_name, vectordb, question)

# question = "What are the new features included in release 1.20?"
# retrieval_qa(llm_name, vectordb, question)

# question = "What are the VAPT issues have been resolved in this ZorroSign release 1.20?"
# retrieval_qa(llm_name, vectordb, question)

