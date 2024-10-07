from llm.llm_chatbot import LLMChatbot
from config_loader import config_loader
from streamline import run_streamline_process  

# Load configurations (if necessary)
llm_config = config_loader.get_llm_config()

# Initialize the LLMChatbot once
chatbot = LLMChatbot(
    model_name=llm_config.get('model_name', 'mistral'),
    temperature=llm_config.get('temperature', 0.7),
    max_tokens=llm_config.get('max_tokens', 1024),
    api_url=llm_config.get('api_url', 'http://localhost:11434/api/generate')
)

# Example of running the streamline process
query = "Example query to process"
final_result = run_streamline_process(query, chatbot)

print(final_result)  # Or return the result in the API