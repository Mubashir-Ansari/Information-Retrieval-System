# Contains functions that deal with the extraction of documents from a text file (see PR01)

import json
import re
from document import Document

def remove_punctuation(terms):
    return [re.sub(r'[^\w\s]', '', term) for term in terms]

def extract_collection(source_file_path: str) -> list[Document]:
    """
    Loads a text file (aesopa10.txt) and extracts each of the listed fables/stories from the file.
    :param source_file_name: File name of the file that contains the fables
    :return: List of Document objects
    """
    catalog = []  # This dictionary will store the document raw_data.

    # TODO: Implement this function. (PR02)
    try:
        with open(source_file_path, 'r', encoding='utf-8') as file:
            lines = file.read()

        lines = lines.split('\n\n\n')
        for i in range(len(lines) - 1):
            if lines[i].startswith('\n') and not lines[i].startswith('\n\n'):
                document = Document()
                document.document_id = len(catalog)
                document.title = lines[i][1:].strip()
                document.raw_text = lines[i + 1].replace('\n', ' ')
                document.terms = remove_punctuation(document.raw_text.lower().split())
                catalog.append(document)
    
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

    return catalog


def save_collection_as_json(collection: list[Document], file_path: str) -> None:
    """
    Saves the collection to a JSON file.
    :param collection: The collection to store (= a list of Document objects)
    :param file_path: Path of the JSON file
    """

    serializable_collection = []
    for document in collection:
        serializable_collection += [{
            'document_id': document.document_id,
            'title': document.title,
            'raw_text': document.raw_text,
            'terms': document.terms,
            'filtered_terms': document.filtered_terms,
            'stemmed_terms': document.stemmed_terms,
            'stemmed_filtered_terms':document.stemmed_filtered_terms
        }]

    with open(file_path, "w") as json_file:
        json.dump(serializable_collection, json_file)


def load_collection_from_json(file_path: str) -> list[Document]:
    """
    Loads the collection from a JSON file.
    :param file_path: Path of the JSON file
    :return: list of Document objects
    """
    try:
        with open(file_path, "r") as json_file:
            json_collection = json.load(json_file)

        collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get('document_id')
            document.title = doc_dict.get('title')
            document.raw_text = doc_dict.get('raw_text')
            document.terms = doc_dict.get('terms')
            document.filtered_terms = doc_dict.get('filtered_terms')
            document.stemmed_terms = doc_dict.get('stemmed_terms')
            document.stemmed_filtered_terms=doc_dict.get('stemmed_filtered_terms')
            collection += [document]

        return collection
    except FileNotFoundError:
        print('No collection was found. Creating empty one.')
        return []
