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
        dataset_links = extract_dataset_links(soup)  # Extracts format per dataset
        publisher = extract_publisher(soup)
        topic = extract_topic(soup)

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

def extract_topic(soup):
    topic_label = soup.find('dt', text='Topic:')
    if topic_label:
        topic_element = topic_label.find_next('dd')
        if topic_element:
            return topic_element.text.strip()
    return "Unknown Topic"

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
    datasets = scrape_datasets(yaml_path)
    save_to_csv(datasets, output_file)

if __name__ == "__main__":
    datasets = scrape_datasets('config/data.yaml')
    save_to_csv(datasets)
