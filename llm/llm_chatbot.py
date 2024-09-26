# llm/llm_chatbot.py

import os
import requests
import json
import logging
from typing import Optional
from config_loader import config_loader

logger = logging.getLogger(__name__)

class LLMChatbot:
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, api_url: Optional[str] = None):
        llm_config = config_loader.get_llm_config()
        
        self.model_name = model_name or llm_config.get('model_name', 'mistral')
        self.temperature = temperature if temperature is not None else llm_config.get('temperature', 0.7)
        self.max_tokens = max_tokens if max_tokens is not None else llm_config.get('max_tokens', 512)
        self.api_url = api_url or llm_config.get('api_url', 'http://localhost:11434/api/generate')
        
        self.session = requests.Session()
        logger.info(f"Initialized LLMChatbot with model: {self.model_name}, temperature: {self.temperature}, max_tokens: {self.max_tokens}, api_url: {self.api_url}")
    
    def __del__(self):
        self.session.close()
        logger.info("Closed LLMChatbot session.")
    
    def generate_response(self, context: str, query: str) -> str:
        """
        Generate a response from the LLM based on the context and query.
    
        Args:
            context (str): Contextual information.
            query (str): The actual prompt or question.
    
        Returns:
            str: The LLM's generated response.
        """
        prompt = f"{context}\n\n{query}"
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        logger.debug(f"Sending payload to LLM: {payload}")

        try:
            response = self.session.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()

            final_answer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            final_answer += chunk['response']
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode chunk: {line}")

            logger.debug(f"Received response from LLM: {final_answer}")
            return final_answer.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return "Error: Failed to retrieve a response from the LLM."

   