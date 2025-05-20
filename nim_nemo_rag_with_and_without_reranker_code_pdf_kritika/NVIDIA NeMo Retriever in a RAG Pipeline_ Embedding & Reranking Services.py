import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_nvidia_ai_endpoints.embeddings import NeMoEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints.reranking import NVIDIARerank
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Step 1: Load .env file (recommended way to handle secrets)
load_dotenv()

# Step 2: Ensure NVIDIA API key is loaded correctly
api_key = os.getenv('NVIDIA_API_KEY')
if not api_key:
    raise ValueError("NVIDIA_API_KEY is missing. Please set it in your .env file or environment.")


# Connect to NeMo Retriever Embedding service (local deployment)
embedding_model = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    api_key=api_key,
    # embedding model ID to use (1B param Llama-based QA embedder)
    # base_url="http://localhost:8080/v1", # base URL of the embedding NIM service (OpenAI-like API)
    batch_size=16 # batch size for embedding calls (tune based on your GPU)
)


# Connect to NeMo Retriever Reranking service (cloud NGC)
reranker = NVIDIARerank(
    model="nvidia/nv-rerankqa-mistral-4b-v3", # reranker model ID (3.5B param Mistral model fine-tuned for QA)
    api_key=api_key,
    # base_url="http://localhost:8000/v1"
    # base URL for the reranking NIM (local service)
    # If using hosted API: provide nvidia_api_key instead of base_url
)


# Example document texts to index (in practice, load and chunk your actual data)
docs = ["NVIDIA H200 is the first GPU with 141 GB of HBM3e memory, delivering 4.8TB/s of bandwidth.","The NVIDIA Triton Inference Server supports multiple frameworks and provides an HTTP/GRPC endpoint for AI model serving.",
# ... (more documents)
]
# Create a FAISS vector store by embedding all documents via the NeMo embedding service
vector_store = FAISS.from_texts(docs, embedding_model)

query = "What interfaces does NVIDIA Triton support?"

# Step 1: Initial retrieval using vector similarity search
candidate_docs = vector_store.similarity_search(query, k=10)
print(f"\n\nInitial documents retrieved (without re-ranker) : {len(candidate_docs)}")

# Step 2: Neural reranking of the candidates
reranked_docs = reranker.compress_documents(documents=candidate_docs,
query=query)
print(f"Reranked top documents: {len(reranked_docs)}")

for doc in reranked_docs:
    score = doc.metadata.get("relevance_score", None)
    snippet = doc.page_content[:100] # first 100 characters of the doc
    print(f"Score: {score:.3f} | Passage: {snippet}...")

# This might output something like:

# Score: 16.625 | Passage: NVIDIA H200 Tensor Core GPU | Datasheet 1 – NVIDIA H200
# Tensor Core GPU supercharges...
# Score: 11.508 | Passage: NVIDIA H200 NVL is the ideal choice for customers with
# space constraints...
# Score: 8.258 | Passage: NVIDIA H200 Tensor Core GPU | Datasheet 2 – Memory
# bandwidth is crucial for HPC applications...



# Initialize an LLM (for example, a 70B Llama2 model hosted by NVIDIA API or a local NIM)
llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct",
    max_tokens=500,
    nvidia_api_key= api_key
)
# Construct a prompt with the top reranked context
# context_text = "\n".join([doc.page_content for doc in reranked_docs])
# prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
# response = llm.invoke(prompt)
# print(response.content)

# Use LangChain prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant. Answer the user's question based on the given context, in just ONE WORD.\n"
     "If the answer is not in the context, say 'I don't know.'\n"
     "Context:\n{context}"),
    ("user", "{question}")
])

formatted_prompt = prompt_template.format_prompt(
    context="\n".join([doc.page_content for doc in reranked_docs]),
    question=query
)

response = llm.invoke(formatted_prompt.to_messages())
print(response.content)



