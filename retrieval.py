import os
import openai
import sys
from datetime import datetime
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

from enum import Enum

class Color(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

def printc(color:Color, text):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"\n {color.value}{current_time} {text} {Color.RESET.value}")

def pretty_print_docs(color, docs):
    printc(Color.GREEN, f"{color}\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    printc(Color.MAGENTA, f"\n{'-' * 100}\n")




embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

printc(Color.RED, ("vectordb collection count: ", vectordb._collection.count()))

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smalldb = Chroma.from_texts(texts, embedding=embedding)
question = "Tell me about all-white mushrooms with large fruiting bodies"


for doc in smalldb.similarity_search(question, k=2):
    print("similarity_search - Page Content: ", doc.page_content)

for doc in smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3):
    print("max_marginal_relevance_search: ", doc.page_content)

question = "what are the advantages of Distributed Tracing?"
docs_ss = vectordb.similarity_search(question,k=3)

printc(Color.YELLOW, f"Question: {question}")
print("similarity_search: ", docs_ss[0].page_content[:100])
print("similarity_search: ", docs_ss[1].page_content[:100])

docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
print("max_marginal_relevance_search: ", docs_mmr[0].page_content[:100])
print("max_marginal_relevance_search: ", docs_mmr[1].page_content[:100])

question = "what are the main features included in release 1.20?"
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/Release Notes for Version 1.20 R1.pdf"}
)

printc(Color.YELLOW, f"\n Question: {question}")
print("Answers from specific source: docs/Release Notes for Version 1.20 R1.pdf")
for d in docs:
    print(d.metadata, d.page_content)


# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The case studies, should be one of `docs/Ant Group (A) 122003-PDF-ENG.pdf` or `docs/Digital Innovation Slides.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the case studies",
        type="integer",
    ),
]

document_content_description = "Case studies"
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)


def search_documents(question, retriever):
    print()
    printc(Color.YELLOW, f"question: {question}")
    docs = retriever.invoke(question)
    for d in docs:
        printc(Color.GREEN, (d.metadata, d.page_content))
    printc(Color.MAGENTA, f"\n{'=' * 100}\n")

question = "What is Ant Group Mission Statement?"
search_documents(question, retriever)


question = "What is the pioneered concept of Sephora?"
search_documents(question, retriever)


question = "What is Pocket Contour?"
search_documents(question, retriever)

question = "what human qualities will still be useful in the future?"
search_documents(question, retriever)


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Wrap our vectorstore
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

question = "what Ant was built on?"
printc(Color.YELLOW, question)
compressed_docs = compression_retriever.invoke(question)
pretty_print_docs(GREEN, compressed_docs)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)

question = "What were the emerging in the online learning market?"
printc(Color.YELLOW, question)
compressed_docs = compression_retriever.invoke(question)
pretty_print_docs(GREEN, compressed_docs)
