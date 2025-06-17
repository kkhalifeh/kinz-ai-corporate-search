import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Initialize Pinecone client
pc = Pinecone()

# Define index name and parameters
index_name = "kinz-classification"
dimension = 3072  # text-embedding-3-large has 3072 dimensions
metric = "cosine"

# Check if the index exists, create it if it doesn't
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Replace with your Pinecone region
        )
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(5)
print(f"Using Pinecone index '{index_name}'")

# Process each level of classification
levels = [
    ("sectors_updated.csv", "sector"),
    ("subsectors_updated.csv", "subsector"),
    ("groups_updated.csv", "group"),
    ("filters_updated.csv", "filter")
]

for file_path, level in levels:
    print(f"Processing {level} data from {file_path}...")
    df = pd.read_csv(file_path, encoding='utf-8')

    # Prepare documents for LangChain
    documents = []
    for _, row in df.iterrows():
        combined_name = row["combined_name"]

        # Prepare metadata based on level
        metadata = {"level": level, "combined_name": combined_name}
        if level == "sector":
            metadata["id"] = str(row["id"])
        elif level == "subsector":
            metadata["id"] = str(row["id"])
            metadata["sector_id"] = str(row["sector_id"])
        elif level == "group":
            metadata["id"] = str(row["id"])
            metadata["subsector_id"] = str(row["subsector_id"])
        elif level == "filter":
            metadata["id"] = str(row["id"])
            metadata["group_id"] = str(row["group_id"])
            metadata["is_searchable"] = bool(row["is_searchable"])

        # Create a Document object for LangChain
        doc = Document(page_content=combined_name, metadata=metadata)
        documents.append(doc)

    # Upload to Pinecone in smaller batches
    batch_size = 50  # Reduced batch size to stay under 4MB limit
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            PineconeVectorStore.from_documents(
                batch,
                embeddings,
                index_name=index_name,
                namespace=level
            )
            print(
                f"Uploaded batch {i // batch_size + 1} for {level} ({len(batch)} documents)")
        except Exception as e:
            print(
                f"Error uploading batch {i // batch_size + 1} for {level}: {e}")
            # Optionally retry with smaller batch size or log for debugging
            continue
        time.sleep(1)  # Avoid rate limiting

    print(f"Uploaded {level} data to Pinecone")

print("All embeddings uploaded to Pinecone.")
