import requests
from bs4 import BeautifulSoup
import yaml
import pandas as pd
import os

def scrape_datasets(yaml_path='data.yaml'):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    urls = [dataset['url'] for dataset in data['datasets']]
    
    all_datasets = []
    for url in urls:
        datasets = scrape_data_from_url(url)
        all_datasets.extend(datasets)
    
    return all_datasets

def scrape_data_from_url(url):
    datasets = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        summary = extract_summary(soup)
        dataset_title = extract_dataset_title(soup)
        dataset_links = extract_dataset_links(soup)

        for dataset in dataset_links:
            dataset_content = download_and_extract_dataset(dataset['link'])
            if dataset_content:
                datasets.append({
                    "title": dataset_title, 
                    "summary": summary, 
                    "name": dataset['name'],
                    "links": dataset['link'], 
                    "content": dataset_content
                })

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

    return datasets

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

if __name__ == "__main__":
    datasets = scrape_datasets('data.yaml')
    save_to_csv(datasets) 