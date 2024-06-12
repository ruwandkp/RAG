import os
import openai
import sys
import os
sys.path.append('../..')
from colorPrint import printc, Color
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#import panel as pn  # GUI
#pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
llm_name = "gpt-3.5-turbo"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "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"


persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

question = "What are the special notes known issues included in release 1.20?"
docs = vectordb.similarity_search(question,k=10)
printc(Color.YELLOW,f"Vector DB docs length: {len(docs)}") 

llm = ChatOpenAI(model_name=llm_name, temperature=0)
printc(Color.YELLOW, f"Chat prediction: {llm.invoke("Hello world!")}")

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
question = "What are the known issues in release 1.20?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

printc(Color.YELLOW, f"Questio: {question}")
result = qa_chain.invoke({"query": question})

printc(Color.GREEN, f"Result: {result["result"]}")