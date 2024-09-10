import requests
import json

class LLMChatbot:
    def __init__(self, model_name='mistral'):
        self.model_name = model_name
        self.api_url = 'http://localhost:11434/api/generate'  # Local API endpoint for Ollama

    def refine_query_with_faiss_results(self, query, faiss_results):
        """
        Use the LLM to refine the results from the FAISS search.
        
        Args:
            query (str): The original user query.
            faiss_results (list): The results from the FAISS search.
        
        Returns:
            list: Refined FAISS results based on the LLM's processing.
        """
        # Construct a prompt for the LLM to refine the FAISS results
        prompt = (
            f"The user query is: '{query}'.\n\n"
            f"Here are the top results from a search:\n{'\n'.join(faiss_results)}\n\n"
            "Please refine these results and provide a summary of the most relevant information."
        )
        
        # Send the prompt to the LLM and get the response
        final_answer = self.generate_response(context='\n'.join(faiss_results), query=prompt)
        
        # For now, just return the LLM's interpretation of the FAISS results
        return [final_answer.strip()]    

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
    
    def refine_query(self, original_query, text_data):
        prompt = (
            f"The user query is: '{original_query}'.\n"
            f"Here is a sample of the available dataset content:\n{text_data[:5]}\n\n"
            "Please refine this query to be more specific and better suited to the available dataset content."
        )
        refined_query = self.generate_response(context="", query=prompt)
        return refined_query.strip()

    def determine_k_from_query(self, refined_query):
        prompt = (
            f"The refined query is: '{refined_query}'.\n\n"
            "Based on the user's query, how many top results should be returned from the search?\n"
            "Return only a number, indicating how many results are expected."
        )
        k_value = self.generate_response(context="", query=prompt)
        try:
            k = int(k_value.strip())
            return max(1, k)  # Ensure k is at least 1
        except ValueError:
            return 2  # Default to 2 results if parsing fails


    def rank_results(self, query, search_results):
        prompt = (
            f"The user query is: '{query}'.\n"
            f"Here are the top search results:\n{search_results}\n\n"
            "Please rank these results in order of relevance to the query."
        )
        ranked_results = self.generate_response(context="", query=prompt)
        return ranked_results.split('\n')