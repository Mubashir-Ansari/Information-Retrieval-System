# Contains all functions related to the porter stemming algorithm.

from document import Document
from typing import List
import re

def is_vowel(char: str, prev_char: str = '') -> bool:
    return char in 'aeiou' or (char == 'y' and prev_char not in 'aeiou')

def get_measure(term: str) -> int:
    """
    Returns the measure m of a given term [C](VC){m}[V].
    :param term: Given term/word
    :return: Measure value m
    """
    measure = 0
    in_vowel_sequence = False

    for i, char in enumerate(term):
        if is_vowel(char, term[i-1] if i > 0 else ''):
            in_vowel_sequence = True
        elif in_vowel_sequence:
            measure += 1
            in_vowel_sequence = False

    return measure


def condition_v(stem: str) -> bool:
    """
    Returns whether condition *v* is true for a given stem (= the stem contains a vowel).
    :param stem: Word stem to check
    :return: True if the condition *v* holds
    """
    for i, char in enumerate(stem):
        if is_vowel(char, stem[i-1] if i > 0 else ''):
            return True
    return False


def condition_d(stem: str) -> bool:
    """
    Returns whether condition *d is true for a given stem (= the stem ends with a double consonant (e.g. -TT, -SS)).
    :param stem: Word stem to check
    :return: True if the condition *d holds
    """
    return len(stem) > 1 and stem[-1] == stem[-2] and not is_vowel(stem[-1])


def cond_o(stem: str) -> bool:
    """
    Returns whether condition *o is true for a given stem (= the stem ends cvc, where the second c is not W, X or Y
    (e.g. -WIL, -HOP)).
    :param stem: Word stem to check
    :return: True if the condition *o holds
    """
    if len(stem) < 3:
        return False
    return (
        not is_vowel(stem[-1]) and
        is_vowel(stem[-2], stem[-3]) and
        not is_vowel(stem[-3]) and
        stem[-1] not in 'wxy'
    )


def stem_term(term: str) -> str:
    """
    Stems a given term of the English language using the Porter stemming algorithm.
    :param term:
    :return:
    """
    # TODO: Implement this function. (PR03)
    # Note: See the provided file "porter.txt" for information on how to implement it!
    step1_suffixes = [
        ("sses", "ss"),
        ("ies", "i"),
        ("ss", "ss"),
        ("s", "")
    ]
    step2_suffixes = [
        ("tional", "tion"),
        ("ational", "ate"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("xflurti","xti")
    ]
    step3_suffixes = [
        ('icate', 'ic'),
        ('ative', ''),
        ('alize', 'al'),
        ('iciti', 'ic'),
        ('ical', 'ic'),
        ('ful', ''),
        ('ness', '')
    ]
    step4_suffixes = [
        'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize'
    ]

    # Step 1a
    for suffix, replacement in step1_suffixes:
        if term.endswith(suffix):
            term = term[:-len(suffix)] + replacement
            break

    # Step 1b
    for suffix, replacement in [("eed", "ee"), ("ed", ""), ("ing", "")]:
        if term.endswith(suffix):
            stem = term[:-len(suffix)]
            if condition_v(stem):
                term = stem + replacement
                if term.endswith("at") or term.endswith("bl") or term.endswith("iz"):
                    term += "e"
                elif condition_d(term) and not (term.endswith("l") or term.endswith("s") or term.endswith("z")):
                    term = term[:-1]
                elif get_measure(term) == 1 and cond_o(term):
                    term += "e"
                break

    # Step 1c
    if term.endswith("y"):
        stem = term[:-1]
        if condition_v(stem):
            term = stem + "i"

    # Step 2
    if get_measure(term) > 0 and term.endswith('flurti'):
        term = term[:-6] + 'ti'

    # Step 2 continued (excluding 'logi')
    for suffix, replacement in step2_suffixes:
        if term.endswith(suffix):
            stem = term[:-len(suffix)]
            if get_measure(stem) > 0:
                term = stem + replacement
                break
    # Step 3
    for suffix, replacement in step3_suffixes:
        if term.endswith(suffix):
            stem = term[:-len(suffix)]
            if get_measure(stem) > 0:
                term = stem + replacement
                break
    # Step 4
    for suffix in step4_suffixes:
        if term.endswith(suffix) and get_measure(term[:-len(suffix)]) > 1:
            term = term[:-len(suffix)]
            break

    # Step 5a
    if term.endswith('e'):
        a = get_measure(term[:-1])
        if a > 1 or (a == 1 and not cond_o(term[:-1])):
            term = term[:-1]

    # Step 5b
    if condition_d(term) and get_measure(term) > 1:
        term = term[:-1]

    return term

def stem_all_documents(collection: list[Document]):
    """
    For each document in the given collection, this method uses the stem_term() function on all terms in its term list.
    Warning: The result is NOT saved in the document's term list, but in the extra field stemmed_terms!
    :param collection: Document collection to process
    """
    try:
        for doc in collection:
            doc.stemmed_terms = [stem_term(term) for term in doc.terms if stem_term(term) is not None]
            # If stopword filtering was already performed, update stemmed_filtered_terms
            if hasattr(doc, 'filtered_terms'):
                doc.stemmed_filtered_terms = [stem_term(term) for term in doc.filtered_terms if stem_term(term) is not None]
    except Exception as e:
        print(f"Error processing document collection: {e}")

def stem_query_terms(query: str) -> str:
    """
    Stems all terms in the provided query string.
    :param query: User query, may contain Boolean operators and spaces.
    :return: Query with stemmed terms
    """
    try:
        terms = query.split()
        stemmed_terms = [stem_term(term) for term in terms if stem_term(term) is not None]
        return " ".join(stemmed_terms)
    except Exception as e:
        print(f"Error stemming query terms '{query}': {e}")
        return query
