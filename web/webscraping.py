import requests
from bs4 import BeautifulSoup
import yaml
import pandas as pd
import os
from llm.llm_chatbot import LLMChatbot
from config_loader import config_loader
import logging

logger = logging.getLogger(__name__)

def extract_topic(soup, title, summary, chatbot: LLMChatbot):
    """
    Extract the topic metadata from the dataset's HTML. If the extracted topic is missing 
    or not in the predefined list, use the LLM to generate a fitting topic based on the 
    dataset's title and summary.
    """
    # Predefined valid topics
    predefined_topics = [
        "Business and economy", "Crime and justice", "Defence", "Education", 
        "Environment", "Government", "Government spending", "Health", 
        "Mapping", "Society", "Towns and cities", "Transport", 
        "Digital service performance", "Government reference data"
    ]

    # Check for a 'Topic:' label in the dataset HTML
    topic_label = soup.find('dt', text='Topic:')
    if topic_label:
        topic_element = topic_label.find_next('dd')
        if topic_element:
            extracted_topic = topic_element.text.strip()

            # If the extracted topic is not in the predefined list, use the LLM
            if extracted_topic not in predefined_topics:
                return generate_llm_topic(title, summary, chatbot)

            # If the topic is valid, return it
            return extracted_topic

    # If no topic is found, use the LLM to infer the topic
    return generate_llm_topic(title, summary, chatbot)

def generate_llm_topic(title, summary, chatbot: LLMChatbot) -> str:
    """
    Use the LLM to generate a suitable topic if the extracted topic is missing or invalid.
    The LLM will pick a topic from the predefined list.
    """
    predefined_topics = [
        "Business and economy", "Crime and justice", "Defence", "Education", 
        "Environment", "Government", "Government spending", "Health", 
        "Mapping", "Society", "Towns and cities", "Transport", 
        "Digital service performance", "Government reference data"
    ]

    # Create the prompt for the LLM
    prompt = (
        f"Based on the following dataset title and summary, choose the most fitting topic from the predefined list:\n\n"
        f"Title: {title}\n"
        f"Summary: {summary}\n\n"
        f"Predefined topics: {', '.join(predefined_topics)}\n\n"
        "Please select the most appropriate topic for this dataset."
    )

    # Generate the LLM response
    try:
        llm_response = chatbot.generate_response(context=summary, query=prompt)
        logger.info(f"LLM topic generation response: {llm_response}")

        # Ensure the LLM chooses from the predefined topics (simple check)
        for topic in predefined_topics:
            if topic.lower() in llm_response.lower():
                return topic

        # If no valid topic was found, fallback to "Unknown Topic"
        return "Unknown Topic"
    
    except Exception as e:
        logger.error(f"Failed to generate topic using LLM: {e}")
        return "Unknown Topic"
def scrape_datasets(yaml_path='config/data.yaml', chatbot: LLMChatbot = None):
    if chatbot is None:
        raise ValueError("chatbot instance is required for topic generation.")

    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    urls = [dataset['url'] for dataset in data['datasets']]
    
    all_datasets = []
    for url in urls:
        datasets = scrape_metadata_from_url(url, chatbot)  # Pass chatbot to scrape_metadata_from_url
        all_datasets.extend(datasets)
    
    return all_datasets

def scrape_metadata_from_url(url, chatbot: LLMChatbot):
    datasets = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        summary = extract_summary(soup)
        dataset_title = extract_dataset_title(soup)
        dataset_links = extract_dataset_links(soup)  # Extracts format per dataset
        publisher = extract_publisher(soup)
        topic = extract_topic(soup, dataset_title, summary, chatbot)  # Use chatbot for missing topic

        for dataset in dataset_links:
            # Clean up the dataset name by removing unwanted text
            cleaned_name = dataset['name'].strip()  # We only keep the visible name
            
            datasets.append({
                "title": dataset_title, 
                "summary": summary, 
                "name": cleaned_name,  # Cleaned name (visible text only)
                "links": dataset['link'],  
                "publisher": publisher,
                "topic": topic,
                "format": dataset['format']  # Correct format for each dataset
            })

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

    return datasets

def extract_publisher(soup):
    publisher_label = soup.find('dt', text='Published by:')
    if publisher_label:
        publisher_element = publisher_label.find_next('dd')
        if publisher_element:
            return publisher_element.text.strip()
    return "Unknown Publisher"

def extract_dataset_links(soup):
    """
    Extract dataset links and their corresponding formats from the table.
    Each row in the table represents a dataset with its own download link and format.
    """
    links = []
    
    # Find all rows in the table body
    rows = soup.find_all('tr', class_='govuk-table__row')
    
    for row in rows:
        # Extract the download link and dataset name from the first <td> cell
        dataset_cell = row.find('td', class_='govuk-table__cell')
        if dataset_cell:
            link_element = dataset_cell.find('a', class_='govuk-link', attrs={'data-ga-event': 'download'})
            if link_element:
                link = link_element['href']

                # Extract the visible text (ignore anything in <span class="visually-hidden">)
                dataset_name = ''.join(link_element.find_all(text=True, recursive=False)).strip()

                # Extract the format from the second <td> cell in the same row
                format_cell = row.find_all('td', class_='govuk-table__cell')[1]
                file_format = format_cell.text.strip() if format_cell else "Unknown Format"

                # Add the dataset info to the list
                links.append({
                    "name": dataset_name,
                    "link": link,
                    "format": file_format
                })
    
    return links

def download_and_extract_dataset(link):
    try:
        response = requests.get(link)
        response.raise_for_status()
        
        if not os.path.exists('temp'):
            os.makedirs('temp')

        filename = os.path.join("temp", os.path.basename(link))
        with open(filename, 'wb') as file:
            file.write(response.content)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.endswith('.json'):
            df = pd.read_json(filename)
        else:
            print(f"Unsupported file format: {filename}")
            return None
        
        # Add a colon after each column name
        df.columns = [f"{col}:" for col in df.columns]

        dataset_content = df.to_string(index=False)
        return dataset_content
    
    except Exception as e:
        print(f"Failed to download or process {link}: {e}")
        return None

def extract_summary(soup):
    summary_element = soup.find('div', class_='js-summary')
    if summary_element:
        p_tag = summary_element.find('p')
        if p_tag:
            return p_tag.text.strip()
    return "No summary available"

def extract_dataset_title(soup):
    title_element = soup.find('h1', class_='heading-large', attrs={'property': 'dc:title'})
    return title_element.text.strip() if title_element else "Unnamed Dataset"

def save_to_csv(datasets, filename='datasets.csv'):
    df = pd.DataFrame(datasets)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def run_webscraping(yaml_path='config/data.yaml', output_file='datasets.csv'):
    # Load the LLM chatbot configuration
    llm_config = config_loader.get_llm_config()
    chatbot = LLMChatbot(
        model_name=llm_config.get('model_name', 'mistral'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 1024),
        api_url=llm_config.get('api_url', 'http://localhost:11434/api/generate')
    )

    datasets = scrape_datasets(yaml_path, chatbot=chatbot)
    save_to_csv(datasets, output_file)

if __name__ == "__main__":
    run_webscraping('config/data.yaml')
