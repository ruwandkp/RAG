import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


print(openai.api_key)

#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs\\Release Notes for Version 1.20 R1.pdf")
pages = loader.load()

print(len(pages))

#page = pages[0]
#for page in pages:
 #   print(page.page_content[0:])
 #   print(page.metadata)

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

chunk_size =5
chunk_overlap = 2

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator=' '
)

text3 = "a bcde f g h i j k l m n o p q r s t u v w x y z"
print(text3)
print(r_splitter.split_text(text3))
print(c_splitter.split_text(text3))

some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
r_splitter.split_text(some_text)

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(some_text)

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)
print(docs)

