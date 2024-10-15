import requests
from bs4 import BeautifulSoup
import yaml
import pandas as pd
import os

def scrape_datasets(yaml_path='config/data.yaml'):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    urls = [dataset['url'] for dataset in data['datasets']]
    
    all_datasets = []
    for url in urls:
        datasets = scrape_metadata_from_url(url)
        all_datasets.extend(datasets)
    
    return all_datasets

def scrape_metadata_from_url(url):
    datasets = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        summary = extract_summary(soup)
        dataset_title = extract_dataset_title(soup)
        dataset_links = extract_dataset_links(soup)
        publisher = extract_publisher(soup)
        topic = extract_topic(soup)
        dataset_format = extract_format(soup)

        for dataset in dataset_links:
            datasets.append({
                "title": dataset_title, 
                "summary": summary, 
                "name": dataset['name'],
                "links": dataset['link'],  # Only link, no content download
                "publisher": publisher,
                "topic": topic,
                "format": dataset_format
            })

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

    return datasets

def extract_publisher(soup):
    # Search for the <dt> tag that contains 'Published by:'
    publisher_label = soup.find('dt', text='Published by:')
    
    if publisher_label:
        # Find the next <dd> tag after the <dt> tag
        publisher_element = publisher_label.find_next('dd')
        if publisher_element:
            return publisher_element.text.strip()
    
    # If no 'Published by:' label is found, return a default value
    return "Unknown Publisher"

def extract_topic(soup):
    # Search for the <dt> tag that contains 'Topic:'
    topic_label = soup.find('dt', text='Topic:')
    
    if topic_label:
        # Find the next <dd> tag after the <dt> tag
        topic_element = topic_label.find_next('dd')
        if topic_element:
            return topic_element.text.strip()
    
    # If no 'Topic:' label is found, return a default value
    return "Unknown Topic"

def extract_format(soup):
    # Look for the <td> tag with class 'govuk-table__cell'
    format_element = soup.find('td', class_='govuk-table__cell')

    if format_element:
        # Clean up the format by stripping extra spaces and unwanted text
        format_text = format_element.text.strip()

        # Extract format by searching for 'Format:' and remove other descriptions
        if "Format:" in format_text:
            # Split by 'Format:' and take only the format, trimming extra text
            format_text = format_text.split("Format:")[-1].split(",")[0].strip()

        # Return the cleaned-up format text
        return format_text
    
    # If no valid format is found, return a default value
    return "Unknown Format"


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

def extract_dataset_links(soup):
    links = []
    link_elements = soup.find_all('a', class_='govuk-link', attrs={'data-ga-event': 'download'})
    
    for element in link_elements:
        link = element['href']

        # The title of the dataset follows the "Download " span
        span_element = element.find('span', class_='visually-hidden')
        if span_element and span_element.text.strip() == 'Download':
            # Extract the name of the dataset which follows the "Download " text
            name = element.text.split('Download ')[-1].split(',')[0].strip()
        else:
            name = "Unknown Dataset"
        
        links.append({"title": name, "link": link, "name": name})
    
    return links

def save_to_csv(datasets, filename='datasets.csv'):
    df = pd.DataFrame(datasets)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def run_webscraping(yaml_path='config/data.yaml', output_file='datasets.csv'):
    datasets = scrape_datasets(yaml_path)
    save_to_csv(datasets, output_file)

if __name__ == "__main__":
    datasets = scrape_datasets('config/data.yaml')
    save_to_csv(datasets) 