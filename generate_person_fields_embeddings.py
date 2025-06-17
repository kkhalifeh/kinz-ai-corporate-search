import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from pinecone import Pinecone
import time
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Initialize Pinecone
index_name = "kinz-classification"
pc = Pinecone()

# Verify the index exists
if index_name not in pc.list_indexes().names():
    print(
        f"Error: Index '{index_name}' does not exist. Please create it first.")
    exit(1)
print(f"Using existing Pinecone index '{index_name}'")

# Process person-related field data
levels = [
    ("title_groups.csv", "title_group"),
    ("genders.csv", "gender"),
    ("salutations.csv", "salutation")
]

for file_path, level in levels:
    print(f"Processing {level} data from {file_path}...")
    df = pd.read_csv(file_path)

    # Handle missing values in combined_name by replacing nan with an empty string
    df["combined_name"] = df["combined_name"].replace(np.nan, "", regex=True)

    # Prepare documents for LangChain
    documents = []
    for _, row in df.iterrows():
        combined_name = row["combined_name"]

        # Skip documents with empty combined_name to avoid issues with embeddings
        if not combined_name:
            print(f"Skipping entry with empty combined_name: {row.to_dict()}")
            continue

        # Prepare metadata
        metadata = {"level": level, "combined_name": combined_name}
        metadata["id"] = str(row["id"])
        if level == "title_group":
            metadata["title"] = row["name_en"]
        elif level == "gender":
            metadata["gender"] = row["name_en"]
        elif level == "salutation":
            metadata["salutation"] = row["name_en"]

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
                namespace=level  # Use namespace to separate levels
            )
            print(
                f"Uploaded batch {i // batch_size + 1} for {level} ({len(batch)} documents)")
        except Exception as e:
            print(
                f"Error uploading batch {i // batch_size + 1} for {level}: {e}")
            continue
        time.sleep(1)  # Avoid rate limiting

    print(f"Uploaded {level} data to Pinecone")

print("All person-related field embeddings uploaded to Pinecone.")
