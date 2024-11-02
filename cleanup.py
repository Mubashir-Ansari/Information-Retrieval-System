# Contains all functions that deal with stop word removal.

from document import Document
import re
import os
import json
from collections import Counter

DATA_PATH = 'data'
def remove_symbols(text_string: str) -> str:
    """
    Removes all punctuation marks and similar symbols from a given string.
    Occurrences of "'s" are removed as well.
    :param text:
    :return:
    """
    # TODO: Implement this function. (PR02)
    cleaned_text = re.sub(r"'s\b", "", text_string)
    cleaned_text = re.sub(r'[.,;:!?\'"-]', '', cleaned_text)
    return cleaned_text


def is_stop_word(term: str, stop_word_list: list[str]) -> bool:
    """
    Checks if a given term is a stop word.
    :param stop_word_list: List of all considered stop words.
    :param term: The term to be checked.
    :return: True if the term is a stop word.
    """
    # TODO: Implement this function. (PR02)
    return term.lower() in stop_word_list



def remove_stop_words_from_term_list(term_list: list[str]) -> list[str]:
    """
    Takes a list of terms and removes all terms that are stop words.
    :param term_list: List that contains the terms
    :return: List of terms without stop words
    """
    # Hint:  Implement the functions remove_symbols() and is_stop_word() first and use them here.
    # TODO: Implement this function. (PR02)
    with open(os.path.join(DATA_PATH, 'stopwords.json'), 'r', encoding='utf-8') as file:
        stop_word_list = json.load(file)
    filtered_terms = []
    for term in term_list:
        cleaned_term = remove_symbols(term)
        # Check if the term is a stop word
        if not is_stop_word(cleaned_term, stop_word_list):
            # If it's not a stop word, remove symbols and append to filtered_terms
            filtered_terms.append(cleaned_term)
    return filtered_terms


def filter_collection(collection: list[Document]):
    """
    For each document in the given collection, this method takes the term list and filters out the stop words.
    Warning: The result is NOT saved in the documents term list, but in an extra field called filtered_terms.
    :param collection: Document collection to process
    """
    # Hint:  Implement remove_stop_words_from_term_list first and use it here.
    # TODO: Implement this function. (PR02)
    with open(os.path.join(DATA_PATH, 'stopwords.json'), 'r', encoding='utf-8') as file:
        stop_word_list = json.load(file)
    for document in collection:
        # Filter stop words from the document's terms
        document.filtered_terms = remove_stop_words_from_term_list(document.terms)
        if hasattr(document, 'stemmed_terms'):
            document.stemmed_filtered_terms = [term for term in document.stemmed_terms if term not in stop_word_list]


def load_stop_word_list(raw_file_path: str) -> list[str]:
    """
    Loads a text file that contains stop words and saves it as a list. The text file is expected to be formatted so that
    each stop word is in a new line, e. g. like englishST.txt
    :param raw_file_path: Path to the text file that contains the stop words
    :return: List of stop words
    """
    # TODO: Implement this function. (PR02)
    with open(raw_file_path, 'r', encoding='utf-8') as file:
        stop_word_list = [line.strip() for line in file]
    return stop_word_list

def create_stop_word_list_by_frequency(collection: list[Document]) -> list[str]:
    """
    Uses the method of J. C. Crouch (1990) to generate a stop word list by finding high and low frequency terms in the
    provided collection.
    :param collection: Collection to process
    :return: List of stop words
    """
    # TODO: Implement this function. (PR02)
    all_terms = []
    for document in collection:
        all_terms.extend(document.terms)

    # Step 2: Count term frequencies in the flattened list
    term_counter = Counter(all_terms)
    total_words = len(all_terms)

    # Step 3: Calculate the thresholds
    min_threshold = 0.01 * total_words  # Terms must appear at least this many times
    max_threshold = 0.1 * total_words   # Terms must appear no more than this many times

    # Step 4: Identify stop words based on the thresholds
    stop_words = []
    for term, freq in term_counter.items():
        if freq < min_threshold or freq > max_threshold:
            stop_words.append(term)

    return stop_words