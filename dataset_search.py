# dataset_search.py

import requests
import logging

def search_datasets(dataset_name):
    """
    Searches Hugging Face Datasets based on provided dataset name.

    Parameters:
        dataset_name (str): The name or keywords for searching datasets.

    Returns:
        list: A list of dataset identifiers that match the query.
    """
    try:
        query = '+'.join(dataset_name.split())
        url = f"https://huggingface.co/api/datasets?search={query}"
        response = requests.get(url)
        response.raise_for_status()
        datasets_info = response.json()

        datasets = [dataset['id'] for dataset in datasets_info]
        return datasets

    except Exception as e:
        logging.error(f"Error searching datasets for '{dataset_name}': {e}")
        return []
