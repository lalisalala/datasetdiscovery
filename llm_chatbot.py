import requests
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class LLMChatbot:
    def __init__(self, model_name='mistral', faiss_index_file='datasets.csv'):
        self.model_name = model_name
        self.api_url = 'http://localhost:11434/api/generate'  # Local API endpoint for Ollama
        
        # Initialize FAISS and load data
        self.model, self.index, self.dataframe = self.initialize_faiss(faiss_index_file)

    def initialize_faiss(self, csv_file):
        # Load the dataset from CSV
        df = pd.read_csv(csv_file)
        documents = df['title'] + ' ' + df['summary'] + ' ' + df['content'].fillna('')
        
        # Initialize the Sentence Transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode documents
        embeddings = model.encode(documents.tolist())
        
        # Create a FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        
        return model, index, df

    def search_faiss(self, query, k=10):
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Perform the search
        distances, indices = self.index.search(np.array(query_embedding), k)
        
        # Retrieve the most relevant documents
        relevant_docs = self.dataframe.iloc[indices[0]]
        
        # Combine titles, summaries, and content from the top k results
        combined_results = " ".join(relevant_docs['title'] + " " + relevant_docs['summary'] + " " + relevant_docs['content'])
        
        return combined_results

    def generate_response(self, context, query, k=5):
        # Perform FAISS search to retrieve relevant dataset content
        relevant_content = self.search_faiss(query, k=k)
        
        # Create a structured prompt by combining FAISS results, context, and the query
        prompt = (
            f"Below are some relevant datasets based on your query:\n\n{relevant_content}\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            "Based on the above datasets and context, provide a detailed response:"
        )
        
        # Setup the payload for the API request to the LLM
        payload = {
            'model': self.model_name,
            'prompt': prompt
        }
        
        # Make the API request to Ollama's local endpoint
        response = requests.post(self.api_url, json=payload, stream=True)
        
        if response.status_code == 200:
            try:
                # Collect all the lines and combine the responses
                combined_response = []
                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():  # Ignore empty lines
                        result = json.loads(line)
                        if 'response' in result:
                            combined_response.append(result['response'])
                
                # Combine all the responses into one string
                return ''.join(combined_response)
            
            except ValueError:
                return "Error decoding JSON response."
        
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Failed to generate response."