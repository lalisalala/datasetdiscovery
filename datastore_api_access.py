import requests
import yaml
import pandas as pd
import os
import hashlib
from llm.llm_chatbot import LLMChatbot
import logging
import time
import re
from html import unescape

logger = logging.getLogger(__name__)

API_KEY = "c535ba42-c70c-4cf0-826c-6a93bc6b1f2c"
BASE_URL = "https://data.london.gov.uk/api/3/action/"

def compute_hash(dataset_ids):
    """
    Compute a hash of the dataset IDs to check if the data is up-to-date.
    """
    hash_object = hashlib.md5()
    hash_object.update("".join(sorted(dataset_ids)).encode('utf-8'))
    return hash_object.hexdigest()

def is_data_uptodate(filename, dataset_ids):
    """
    Check if the local datasets.csv file is up-to-date with the website's dataset IDs.
    """
    if not os.path.exists(filename):
        return False  # File doesn't exist, so it's not up-to-date

    try:
        # Read the current datasets.csv
        df = pd.read_csv(filename)

        # Compare the hash of dataset IDs
        current_hash = compute_hash(dataset_ids)
        stored_hash = None

        # Check if the hash is stored in the datasets.csv metadata
        if "hash" in df.columns and len(df) > 0:
            stored_hash = df["hash"].iloc[0]

        if stored_hash == current_hash:
            logger.info("Datasets.csv is up-to-date.")
            return True

        logger.info("Datasets.csv is outdated.")
        return False

    except Exception as e:
        logger.error(f"Error checking datasets.csv: {e}")
        return False

def fetch_datasets_metadata():
    """
    Fetch metadata for all datasets from the London Datastore API and save them in a list.
    """
    headers = {"Authorization": API_KEY}

    # Step 1: Get all dataset IDs
    try:
        response = requests.get(f"{BASE_URL}package_list", headers=headers)
        response.raise_for_status()
        dataset_ids = response.json().get("result", [])
        logger.info(f"Retrieved {len(dataset_ids)} dataset IDs.")
        return dataset_ids
    except Exception as e:
        logger.error(f"Failed to retrieve dataset IDs: {e}")
        return []

def fetch_metadata_for_dataset(dataset_id, headers):
    """
    Fetch metadata for a single dataset using its ID.
    """
    try:
        response = requests.get(f"{BASE_URL}package_show?id={dataset_id}", headers=headers)
        response.raise_for_status()
        dataset = response.json().get("result", {})
        
        # Extract fields from the dataset
        title = dataset.get("title", "Unnamed Dataset")
        summary = dataset.get("notes", "No summary available")
        publisher = dataset.get("organization", {}).get("title", "Unknown Publisher")
        tags = [tag.get("name", "Unknown Topic") for tag in dataset.get("tags", [])]
        resources = dataset.get("resources", [])
        
        # Extract resources (links and formats)
        dataset_resources = []
        for resource in resources:
            dataset_resources.append({
                "name": resource.get("id", "Unknown Resource"),
                "link": resource.get("url", ""),
                "format": resource.get("format", "Unknown Format"),
            })

        # Return metadata for this dataset
        return {
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "topic": ", ".join(tags) if tags else "Unknown Topic",
            "resources": dataset_resources,
        }

    except Exception as e:
        logger.error(f"Error fetching metadata for dataset {dataset_id}: {e}")
        return None

def save_to_csv(datasets, filename, dataset_ids):
    """
    Save the fetched dataset metadata to a CSV file, including a hash of the dataset IDs.
    """
    rows = []
    for dataset in datasets:
        for resource in dataset["resources"]:
            # Clean and process fields
            summary = clean_html(dataset["summary"])
            topics = clean_topics(dataset["topic"])
            publisher = normalize_case(dataset["publisher"])
            title = normalize_case(dataset["title"])
            file_format = extract_file_type(resource["format"])

            rows.append({
                "title": title,
                "summary": summary,
                "publisher": publisher,
                "topic": topics,
                "name": resource["name"],
                "links": resource["link"],
                "format": file_format,
            })

    # Add a hash of dataset IDs to the first row for future comparison
    hash_value = compute_hash(dataset_ids)
    if rows:
        rows[0]["hash"] = hash_value

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    logger.info(f"Cleaned data saved to {filename}")

def run_datastore_access(output_file='datasets.csv'):
    """
    Main function to fetch and save datasets from the London Datastore API.
    """
    logger.info("Checking if datasets.csv is up-to-date...")

    # Step 1: Fetch dataset IDs
    dataset_ids = fetch_datasets_metadata()
    if not dataset_ids:
        logger.error("Failed to fetch dataset IDs.")
        return

    # Step 2: Check if datasets.csv is up-to-date
    if is_data_uptodate(output_file, dataset_ids):
        logger.info("Using existing datasets.csv.")
        return  # Skip fetching and reuse existing file

    # Step 3: Fetch new metadata and update datasets.csv
    logger.info("Fetching new metadata and updating datasets.csv.")
    datasets = []
    headers = {"Authorization": API_KEY}

    for idx, dataset_id in enumerate(dataset_ids):
        logger.info(f"Fetching metadata for dataset {idx + 1}/{len(dataset_ids)}: ID {dataset_id}")
        dataset_metadata = fetch_metadata_for_dataset(dataset_id, headers)
        if dataset_metadata:
            datasets.append(dataset_metadata)
        time.sleep(0.1)  # Delay to avoid rate limiting

    if datasets:
        save_to_csv(datasets, output_file, dataset_ids)
    else:
        logger.warning("No datasets retrieved during the update.")

def clean_html(html):
    """
    Remove HTML tags and decode HTML entities.
    """
    text = re.sub(r'<[^>]+>', '', html)  # Remove HTML tags
    return unescape(text.strip())  # Decode HTML entities and strip whitespace

def clean_topics(topics):
    """
    Normalize and clean the topic field by removing duplicates, formatting properly.
    """
    if not topics or topics.lower() == "unknown topic":
        return "Unknown Topic"
    unique_topics = sorted(set([topic.strip().title() for topic in topics.split(',')]))
    return ", ".join(unique_topics)

def normalize_case(text):
    """
    Normalize text to title case (e.g., "london fire brigade" -> "London Fire Brigade").
    """
    return text.title().strip() if text else "Unknown"

def extract_file_type(format_field):
    """
    Normalize file format field.
    """
    format_mapping = {
        "spreadsheet": "Excel Spreadsheet",
        "pdf": "PDF Document",
        "zip": "ZIP Archive",
        "shp": "GIS Shapefile"
    }
    return format_mapping.get(format_field.lower(), format_field.title())

if __name__ == "__main__":
    run_datastore_access()
