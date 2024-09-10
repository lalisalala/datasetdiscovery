import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_data(csv_file='datasets.csv'):
    # Load the dataset from CSV
    df = pd.read_csv(csv_file)
    metadata = df[['title', 'summary', 'links']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    content = df['content']
    return df, metadata, content

def create_faiss_index(metadata, content):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode metadata and content
    metadata_embeddings = model.encode(metadata.tolist())
    content_embeddings = model.encode(content.tolist())
    
    # Create FAISS indices
    dimension = metadata_embeddings.shape[1]
    metadata_index = faiss.IndexFlatL2(dimension)
    content_index = faiss.IndexFlatL2(dimension)
    
    metadata_index.add(np.array(metadata_embeddings))
    content_index.add(np.array(content_embeddings))
    
    return model, metadata_index, content_index, content_embeddings

def search_faiss(query, model, metadata_index, content_index, content_embeddings, k=20):
    query_embedding = model.encode([query])
    
    # Search in metadata index
    metadata_distances, metadata_indices = metadata_index.search(np.array(query_embedding), k)
    
    # Retrieve relevant content indices based on metadata search
    if len(metadata_indices[0]) > 0:
        relevant_indices = metadata_indices[0]
        relevant_content_embeddings = np.array([content_embeddings[idx] for idx in relevant_indices])
        
        # Search in content index using relevant content embeddings
        content_distances, content_indices = content_index.search(relevant_content_embeddings, k)
        
        return metadata_indices[0], metadata_distances[0], content_indices, content_distances
    else:
        return metadata_indices[0], metadata_distances[0], [], []