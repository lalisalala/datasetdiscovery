import requests
import json

class LLMChatbot:
    def __init__(self, model_name='mistral'):
        self.model_name = model_name
        self.api_url = 'http://localhost:11434/api/generate'  # Local API endpoint for Ollama

    def interpret_query(self, query):
        """
        Use the LLM to classify the query into 'single', 'multiple', or 'broad'.
        
        Args:
            query (str): The user's query.
            
        Returns:
            str: 'single', 'multiple', or 'broad' based on the query.
        """
        prompt = (
            f"The user query is: '{query}'.\n\n"
            "Please classify this query as one of the following:\n"
            "- 'single' if it asks for a single dataset,\n"
            "- 'multiple' if it asks for multiple datasets,\n"
            "- 'broad' if it asks for a general or aggregate response.\n\n"
            "Return only one of these three words."
        )
        
        # Send the query to the LLM for interpretation
        payload = {
            'model': self.model_name,
            'prompt': prompt
        }

        try:
            response = requests.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()

            # Process the streamed response
            final_answer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Ignore empty lines
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            final_answer += chunk['response']
                    except json.JSONDecodeError:
                        print(f"Failed to decode chunk: {line}")

            return final_answer.strip()

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return "broad"  # Default to 'broad' if there's an issue

    def generate_response(self, context, query):
        """
        Use the LLM to generate a natural language response to the query based on the context.
        
        Args:
            context (str): Contextual data (metadata and dataset results).
            query (str): The user's query.
            
        Returns:
            str: Generated natural language response.
        """
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Please generate a detailed and accurate response based on the provided context."
        )
        
        # Send the query to the LLM
        payload = {
            'model': self.model_name,
            'prompt': prompt
        }

        try:
            # Send a POST request with stream=True to handle streaming
            response = requests.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()

            # Process the streamed chunks of JSON
            final_answer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Skip empty lines
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            final_answer += chunk['response']
                    except json.JSONDecodeError:
                        print(f"Failed to decode chunk: {line}")
            
            return final_answer.strip()

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return "Error: Failed to retrieve a response from the LLM."
