import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def create_faiss_index(metadata):
    """
    Create a FAISS index using the embeddings of the LLM-generated summaries.
    """
    # Load the pre-trained model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Encode the metadata (summaries) into embeddings
    metadata_embeddings = model.encode(metadata)
    
    # Create a FAISS index for the embeddings
    dimension = metadata_embeddings.shape[1]
    metadata_index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the FAISS index
    metadata_index.add(np.array(metadata_embeddings))
    
    return model, metadata_index