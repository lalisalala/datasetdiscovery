import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Path to the JSON file with dataset metadata
JSON_FILE = "datasets.json"

# Load metadata from JSON
def load_metadata(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    return data


# Prepare FAISS index
def prepare_faiss_index(data, model_name="all-MiniLM-L6-v2"):
    """
    Prepares a FAISS index from the dataset metadata.

    Args:
        data: List of dataset metadata.
        model_name: The name of the SentenceTransformer model to use.

    Returns:
        index: A trained FAISS index.
        id_to_metadata: A mapping of FAISS IDs to dataset metadata.
    """
    # Load sentence embedding model
    model = SentenceTransformer(model_name)
    id_to_metadata = {}
    embeddings = []

    for idx, dataset in enumerate(data):
        # Combine fields to create a single searchable string
        searchable_text = f"{dataset['title']} {dataset['summary']} {dataset['publisher']} {' '.join(dataset['topics'])}"
        embedding = model.encode(searchable_text, convert_to_numpy=True)
        embeddings.append(embedding)
        id_to_metadata[idx] = dataset  # Map FAISS ID to metadata

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings, dtype="float32")

    # Create FAISS index
    dimension = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Add embeddings to the index

    return index, id_to_metadata


# Perform search
def search_index(index, query, id_to_metadata, top_k=5, model_name="all-MiniLM-L6-v2"):
    """
    Performs a FAISS search for a given query.

    Args:
        index: The FAISS index.
        query: The search query.
        id_to_metadata: Mapping of FAISS IDs to metadata.
        top_k: Number of top results to retrieve.
        model_name: The name of the SentenceTransformer model to use.

    Returns:
        results: A list of top-k results with their metadata and similarity scores.
    """
    # Load sentence embedding model
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)

    # Search in the index
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        metadata = id_to_metadata.get(idx, {})
        results.append({
            "metadata": metadata,
            "distance": dist
        })

    return results


# Main function
def main():
    # Load dataset metadata
    data = load_metadata(JSON_FILE)

    # Prepare FAISS index
    print("Preparing FAISS index...")
    index, id_to_metadata = prepare_faiss_index(data)

    # Perform a search
    query = input("Enter your search query: ")
    top_k = int(input("Enter the number of top results to retrieve: "))
    print(f"Searching for: {query}")

    results = search_index(index, query, id_to_metadata, top_k=top_k)

    # Display results
    print("\nTop Results:")
    for idx, result in enumerate(results):
        metadata = result["metadata"]
        print(f"\nResult {idx + 1}:")
        print(f"Title: {metadata['title']}")
        print(f"Summary: {metadata['summary']}")
        print(f"Publisher: {metadata['publisher']}")
        print(f"Topics: {', '.join(metadata['topics'])}")
        print(f"Distance: {result['distance']}")


if __name__ == "__main__":
    main()
