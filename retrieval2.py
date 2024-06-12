from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

def printc(colour, text):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"\n {colour}{current_time} {text}{RESET}")

# Load PDF
loader = PyPDFLoader("docs/Digital Innovation Slides.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
embedding = OpenAIEmbeddings()
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
printc(YELLOW, docs_svm[0])

question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
printc(YELLOW, docs_tfidf[0])