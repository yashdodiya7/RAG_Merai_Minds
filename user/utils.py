import hashlib
import os
from dotenv import load_dotenv
from typing import List

from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Access the API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# Initialize Pinecone
def initialize_pinecone():
    api_key = pinecone_api_key
    pc = Pinecone(api_key=api_key)
    return pc


def create_index(pc, index_name, dimension):
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )


def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


## New code for the RAG
def chunk_text_for_list(text: str, max_chunk_size: int) -> List[str]:
    # Ensure each text ends with a double newline to correctly split paragraphs
    if not text.endswith("\n"):
        text += "\n"
    # Split text into paragraphs
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    # Iterate over paragraphs and assemble chunks
    for paragraph in paragraphs:
        # Check if adding the current paragraph exceeds the maximum chunk size
        if (
                len(current_chunk) + len(paragraph) + 2 > max_chunk_size
                and current_chunk
        ):
            # If so, add the current chunk to the list and start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = ""
        # Add the current paragraph to the current chunk
        current_chunk += paragraph.strip() + " "
    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

    # Apply the chunk_text function to each document in the list
    # return [chunk_text(docs, max_chunk_size)]


EMBEDDINGS = OpenAIEmbeddings(
    api_key=openai_api_key
)


def generate_embeddings(documents: list[any]) -> list[list[float]]:
    embedded = EMBEDDINGS.embed_documents(documents)
    return embedded


def store_vectors_in_pinecone(pc, vectors, ids, metadata):
    index_name = 'chatbot'
    index = pc.Index(index_name)
    index.upsert(vectors=list(zip(ids, vectors)))


def generate_short_id(content: str) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


def combine_vector_and_text(
        documents: list[any], doc_embeddings: list[list[float]]
) -> list[dict[str, any]]:
    data_with_metadata = []

    for doc_text, embedding in zip(documents, doc_embeddings):
        # Convert doc_text to string if it's not already a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        # Generate a unique ID based on the text content
        doc_id = generate_short_id(doc_text)

        # Create a data item dictionary
        data_item = {
            "id": doc_id,
            "values": embedding[0],
            "metadata": {"text": doc_text},  # Include the text as metadata
        }

        # Append the data item to the list
        data_with_metadata.append(data_item)

    return data_with_metadata


def get_query_embeddings(query: str) -> list[float]:
    query_embeddings = EMBEDDINGS.embed_query(query)
    return query_embeddings
