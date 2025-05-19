import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# To read document, in case we have actual files to read like a pdf, etc
from langchain.docstore.document import Document

# To split big texts read from the document into smaller chunks for better embeddings to be created
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
# Step 1: Load .env file (recommended way to handle secrets)
load_dotenv()

# Step 2: Ensure NVIDIA API key is loaded correctly
api_key = os.getenv('NVIDIA_API_KEY')
if not api_key:
    raise ValueError("NVIDIA_API_KEY is missing. Please set it in your .env file or environment.")

# Step 3: Connect to NVIDIA-hosted embedding model
embedding_model = NVIDIAEmbeddings(
    # model="NV-Embed-QA-003",  # Hosted version alias (preferred for API Catalog)
    model="nvidia/llama-3.2-nv-embedqa-1b-v2", # model name of the embedding NIM
    nvidia_api_key=api_key
)

# Connect to NIM for LLM (chat model)
llm_model = ChatNVIDIA(
    model="meta/llama3-70b-instruct", # model name of the LLM NIM
    # base_url="http://localhost:8000/v1" # endpoint URL of LLM service
    nvidia_api_key=api_key,
    max_tokens=500
)


# Example document content (this could be loaded from a PDF, etc.)
doc_text = """
Tiger Analytics is a firm specializing in data science and AI solutions.
One of its teams focuses on Azure cloud-based deployments of AI models.
It leverages NVIDIA's NeMo framework and NIM microservices for scalable AI
inference.
"""
documents = [Document(page_content=doc_text)]


# Split documents into smaller chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = text_splitter.split_documents(documents)
print(f"Number of chunks: {len(doc_chunks)}")


# Extract raw text from chunks
texts = [chunk.page_content for chunk in doc_chunks]
# Embed all the chunks (get a list of vectors)
embeddings = embedding_model.embed_documents(texts)
print(f"Generated {len(embeddings)} embeddings, each of length {len(embeddings[0])}")


# Create a FAISS vector store from our chunks and embeddings
vector_store = FAISS.from_texts(texts, embedding_model)
# (Alternatively: FAISS.from_documents(doc_chunks, embedding=embedding_model))


# Example user question
query = "What does Tiger Analytics use NVIDIA NeMo for?"
# Step 5a: Embed the query to a vector
query_vector = embedding_model.embed_query(query)
# Step 5b: Use the vector store to find similar document chunks
matched_docs = vector_store.similarity_search_by_vector(query_vector, k=2)
# Get the text content of the top matches
context_snippets = [doc.page_content for doc in matched_docs]
print("Retrieved context:", context_snippets)



# Construct a prompt template with placeholders for context and question
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI assistant. Answer the user's question based on the given context. "
     "If the context does not contain the answer, say you don't know. "
     "Context: {context}"),
    ("user", "{question}")
])


# Format the context into the prompt
formatted_prompt = prompt.format_prompt(
    context="\n\n".join(context_snippets),
    question=query
)
# The prompt object can combine with the llm in a chain, but we'll call the llm directly for simplicity:
response = llm_model(formatted_prompt.to_messages())
print("AI Answer:", response.content)





