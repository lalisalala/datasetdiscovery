
import os
import yaml
import logging

class ConfigLoader:
    def __init__(self, config_path: str = 'config/llm_config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at path: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
                self.setup_logging(config.get('logging', {}))
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {e}")

    def setup_logging(self, logging_config: dict):
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO)
        log_file = logging_config.get('file', 'app.log')
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging is configured.")

    def get_llm_config(self) -> dict:
        return self.config.get('llm', {})

    def get_faiss_config(self) -> dict:
        return self.config.get('faiss', {})

# Instantiate a global config loader for easy access across modules
config_loader = ConfigLoader()
