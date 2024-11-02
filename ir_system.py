import json
import os
import time
import re
import cleanup
import extraction
import models
import porter
from document import Document
from collections import defaultdict
import math

# Important paths:
RAW_DATA_PATH = 'raw_data'
DATA_PATH = 'data'
COLLECTION_PATH = os.path.join(DATA_PATH, 'my_collection.json')
STOPWORD_FILE_PATH = os.path.join(DATA_PATH, 'stopwords.json')

# Menu choices:
(CHOICE_LIST, CHOICE_SEARCH, CHOICE_EXTRACT, CHOICE_UPDATE_STOP_WORDS, CHOICE_SET_MODEL, CHOICE_SHOW_DOCUMENT,
 CHOICE_EXIT) = 1, 2, 3, 4, 5, 6, 9
MODEL_BOOL_LIN, MODEL_BOOL_INV, MODEL_BOOL_SIG, MODEL_FUZZY, MODEL_VECTOR = 1, 2, 3, 4, 5
SW_METHOD_LIST, SW_METHOD_CROUCH = 1, 2


class InformationRetrievalSystem(object):
    def __init__(self):
        if not os.path.isdir(DATA_PATH):
            os.makedirs(DATA_PATH)

        # Collection of documents, initially empty.
        try:
            self.collection = extraction.load_collection_from_json(COLLECTION_PATH)
        except FileNotFoundError:
            print('No previous collection was found. Creating empty one.')
            self.collection = []

        # Stopword list, initially empty.
        try:
            with open(STOPWORD_FILE_PATH, 'r') as f:
                self.stop_word_list = json.load(f)
        except FileNotFoundError:
            print('No stopword list was found.')
            self.stop_word_list = []

        self.model = None  # Saves the current IR model in use.
        self.output_k = 5  # Controls how many results should be shown for a query.


    def main_menu(self):
        """
        Provides the main loop of the CLI menu that the user interacts with.
        """
        while True:
            print(f'Current retrieval model: {self.model}')
            print(f'Current collection: {len(self.collection)} documents')
            print()
            print('Please choose an option:')
            print(f'{CHOICE_LIST} - List documents')
            print(f'{CHOICE_SEARCH} - Search for term')
            print(f'{CHOICE_EXTRACT} - Build collection')
            print(f'{CHOICE_UPDATE_STOP_WORDS} - Rebuild stopword list')
            print(f'{CHOICE_SET_MODEL} - Set model')
            print(f'{CHOICE_SHOW_DOCUMENT} - Show a specific document')
            print(f'{CHOICE_EXIT} - Exit')

            action_choice = int(input('Enter choice: '))

            if action_choice == CHOICE_LIST:
                # List documents in CLI.
                if self.collection:
                    for document in self.collection:
                        print(document)
                else:
                    print('No documents.')

            elif action_choice == CHOICE_SEARCH:
                # Read a query string from the CLI and search for it.

                # Determine desired search parameters:
                SEARCH_NORMAL, SEARCH_SW, SEARCH_STEM, SEARCH_SW_STEM = 1, 2, 3, 4
                print('Search options:')
                print(f'{SEARCH_NORMAL} - Standard search (default)')
                print(f'{SEARCH_SW} - Search documents with removed stopwords')
                print(f'{SEARCH_STEM} - Search documents with stemmed terms')
                print(f'{SEARCH_SW_STEM} - Search documents with removed stopwords AND stemmed terms')

                search_mode = int(input('Enter choice: '))
                stop_word_filtering = (search_mode == SEARCH_SW) or (search_mode == SEARCH_SW_STEM)
                stemming = (search_mode == SEARCH_STEM) or (search_mode == SEARCH_SW_STEM)

                # Actual query processing begins here:
                query = input('Query: ')
                if stemming:
                    query = porter.stem_query_terms(query)

                start_time = time.time()
                if isinstance(self.model, models.InvertedListBooleanModel):
                    results = self.inverted_list_search(query, stemming, stop_word_filtering)
                elif isinstance(self.model, models.VectorSpaceModel):
                    results = self.buckley_lewit_search(query, stemming, stop_word_filtering)
                elif isinstance(self.model, models.SignatureBasedBooleanModel):
                    results = self.signature_search(query, stemming, stop_word_filtering)
                else:
                    results = self.basic_query_search(query, stemming, stop_word_filtering)
                end_time = time.time()
               
                processing_time = (end_time - start_time) * 1000
                # Output of results:
                for (score, document) in results:
                    if score > 0:
                        print(f'{score}: {document}')

                # Output of quality metrics:
                print()
                print(f'precision: {self.calculate_precision(query,results)}')
                print(f'recall: {self.calculate_recall(query,results)}')
                print(f'Query processing time: {processing_time:.2f} ms')

            elif action_choice == CHOICE_EXTRACT:
                # Extract document collection from text file.

                raw_collection_file = os.path.join(RAW_DATA_PATH, 'aesopa10.txt')
                self.collection = extraction.extract_collection(raw_collection_file)
                assert isinstance(self.collection, list)
                assert all(isinstance(d, Document) for d in self.collection)

                if input('Should stopwords be filtered? [y/N]: ') == 'y':
                    cleanup.filter_collection(self.collection)

                if input('Should stemming be performed? [y/N]: ') == 'y':
                    porter.stem_all_documents(self.collection)

                extraction.save_collection_as_json(self.collection, COLLECTION_PATH)
                print('Done.\n')

            elif action_choice == CHOICE_UPDATE_STOP_WORDS:
                # Rebuild the stop word list, using one out of two methods.

                print('Available options:')
                print(f'{SW_METHOD_LIST} - Load stopword list from file')
                print(f"{SW_METHOD_CROUCH} - Generate stopword list using Crouch's method")

                method_choice = int(input('Enter choice: '))
                if method_choice in (SW_METHOD_LIST, SW_METHOD_CROUCH):
                    # Load stop words using the desired method:
                    if method_choice == SW_METHOD_LIST:
                        self.stop_word_list = cleanup.load_stop_word_list(os.path.join(RAW_DATA_PATH, 'englishST.txt'))
                        print('Done.\n')
                    elif method_choice == SW_METHOD_CROUCH:
                        self.stop_word_list = cleanup.create_stop_word_list_by_frequency(self.collection)
                        print('Done.\n')

                    # Save new stopword list into file:
                    with open(STOPWORD_FILE_PATH, 'w') as f:
                        json.dump(self.stop_word_list, f)
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SET_MODEL:
                # Choose and set the retrieval model to use for searches.

                print()
                print('Available models:')
                print(f'{MODEL_BOOL_LIN} - Boolean model with linear search')
                print(f'{MODEL_BOOL_INV} - Boolean model with inverted lists')
                print(f'{MODEL_BOOL_SIG} - Boolean model with signature-based search')
                print(f'{MODEL_FUZZY} - Fuzzy set model')
                print(f'{MODEL_VECTOR} - Vector space model')
                model_choice = int(input('Enter choice: '))
                if model_choice == MODEL_BOOL_LIN:
                    self.model = models.LinearBooleanModel()
                elif model_choice == MODEL_BOOL_INV:
                    self.model = models.InvertedListBooleanModel()
                elif model_choice == MODEL_BOOL_SIG:
                    self.model = models.SignatureBasedBooleanModel()
                elif model_choice == MODEL_FUZZY:
                    self.model = models.FuzzySetModel()
                elif model_choice == MODEL_VECTOR:
                    self.model = models.VectorSpaceModel()
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SHOW_DOCUMENT:
                target_id = int(input('ID of the desired document:'))
                found = False
                for document in self.collection:
                    if document.document_id == target_id:
                        print(document.title)
                        print('-' * len(document.title))
                        print(document.raw_text)
                        found = True

                if not found:
                    print(f'Document #{target_id} not found!')

            elif action_choice == CHOICE_EXIT:
                break
            else:
                print('Invalid choice.')

            print()
            input('Press ENTER to continue...')
            print()

    def basic_query_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Searches the collection for a query string. This method is "basic" in that it does not use any special algorithm
        to accelerate the search. It simply calculates all representations and matches them, returning a sorted list of
        the k most relevant documents and their scores.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        query_representation = self.model.query_to_representation(query)
        document_representations = [self.model.document_to_representation(d, stop_word_filtering, stemming)
                                    for d in self.collection]
        scores = [self.model.match(dr, query_representation) for dr in document_representations]
        ranked_collection = sorted(zip(scores, self.collection), key=lambda x: x[0], reverse=True)
        results = ranked_collection
        return results

    def inverted_list_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast Boolean query search for inverted lists.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        if not hasattr(self.model, 'inverted_index') or not self.model.inverted_index:
            for document in self.collection:
                self.model.document_to_representation(document, stop_word_filtering, stemming)

        document_representations = [self.model.document_to_representation(d, stop_word_filtering, stemming)
                                    for d in self.collection]
        query_representation = self.model.query_to_representation(query)
        scores = [self.model.match(dr, query_representation) for dr in document_representations]
        ranked_collection = sorted(zip(scores, self.collection), key=lambda x: x[0], reverse=True)
        results = ranked_collection
        return results
  

    def buckley_lewit_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast query search for the Vector Space Model using the algorithm by Buckley & Lewit.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        # TODO: Implement this function (PR04)
        if not hasattr(self.model, 'inverted_index') or not self.model.inverted_index:
            for document in self.collection:
                self.model.document_to_representation(document, stop_word_filtering, stemming)

        query_vector = self.model.query_to_representation(query)
        DS = defaultdict(float)

        for term, w_qk in query_vector.items():
            if w_qk > 0:
                doc_list = self.model.inverted_index.get(term, [])
                for doc_id, w_dk in doc_list:
                    if doc_id in DS:
                        DS[doc_id] += w_qk * w_dk
                    else:
                        DS[doc_id] = w_qk * w_dk

        top_docs = sorted(DS.items(), key=lambda item: item[1], reverse=True)

        gamma = 10
        top_docs = top_docs[:gamma]

        results = [(self.model.match(query_vector, self.model.document_to_representation(next(doc for doc in self.collection if doc.document_id == doc_id))),
                    next(doc for doc in self.collection if doc.document_id == doc_id)) 
                   for doc_id, score in top_docs]

        return results

    def signature_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast Boolean query search using signatures for quicker processing.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        # TODO: Implement this function (PR04)
        document_representations = [self.model.document_to_representation(d, stop_word_filtering, stemming)
                                    for d in self.collection]
        query_representation = self.model.query_to_representation(query)
        scores = [self.model.match(dr,query_representation) for dr in document_representations]
        ranked_collection = sorted(zip(scores, self.collection), key=lambda x: x[0], reverse=True)
        results = ranked_collection

        return results
    
    def calculate_precision(self, query: str, result_list: list[tuple]) -> float:
        ground_truth = load_ground_truth(os.path.join(RAW_DATA_PATH, 'ground_truth.txt'))
        query_tokens = self.tokenize_query(query)
        result_doc_ids = self.evaluate_query_tokens(query_tokens, ground_truth)

        retrieved_docs = [doc[1].document_id + 1 for doc in result_list if doc[0] > 0]

        if len(retrieved_docs) == 0:
            return 0.0

        true_positives = set(result_doc_ids).intersection(set(retrieved_docs))
        precision = len(true_positives) / len(retrieved_docs)

        return precision

    def calculate_recall(self, query: str, result_list: list[tuple]) -> float:
        ground_truth = load_ground_truth(os.path.join(RAW_DATA_PATH, 'ground_truth.txt'))
        query_tokens = self.tokenize_query(query)
        result_doc_ids = self.evaluate_query_tokens(query_tokens, ground_truth)

        retrieved_docs = [doc[1].document_id + 1 for doc in result_list if doc[0] >0]

        if len(result_doc_ids) == 0:
            return 0.0

        true_positives = set(result_doc_ids).intersection(set(retrieved_docs))
        recall = len(true_positives) / len(result_doc_ids)

        return recall

    def tokenize_query(self, query: str):
        tokens = []
        current_term = ''
        for char in query:
            if char in '&|()-':
                if current_term.strip():
                    tokens.append(current_term.strip())
                    current_term = ''
                tokens.append(char)
            elif char == ' ':
                if current_term.strip():
                    tokens.append(current_term.strip())
                    current_term = ''
            else:
                current_term += char
        if current_term.strip():
            tokens.append(current_term.strip())
        return tokens

    def evaluate_query_tokens(self, tokens, ground_truth):
        operation_stack = []
        operator_stack = []
        operator_precedence = {'|': 1, '&': 2, '-': 3}

        def apply_operation():
            operation = operator_stack.pop()
            if operation == '-':
                term = operation_stack.pop()
                result = set(range(1, max(ground_truth.values(), default=[0])[0] + 1)) - term
                operation_stack.append(result)
            else:
                right = operation_stack.pop()
                left = operation_stack.pop()
                if operation == '&':
                    result = left & right
                elif operation == '|':
                    result = left | right
                operation_stack.append(result)

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    apply_operation()
                operator_stack.pop()
            elif token in operator_precedence:
                while (operator_stack and operator_stack[-1] in operator_precedence and
                    operator_precedence[token] <= operator_precedence[operator_stack[-1]]):
                    apply_operation()
                operator_stack.append(token)
            else:
                term_docs = ground_truth.get(token, set())
                operation_stack.append(term_docs)
            i += 1

        while operator_stack:
            apply_operation()

        return operation_stack[0] if operation_stack else set()


    
def load_ground_truth(filepath) -> dict:
    ground_truth = {}
    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Line must contain the expected separator
                if ' - ' in line:
                    term, doc_ids_str = line.split(' - ')
                    doc_ids = set(map(int, doc_ids_str.split(', ')))
                    ground_truth[term] = doc_ids
                else:
                    raise ValueError(f"Line in file does not conform to expected format: {line}")

    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return ground_truth

if __name__ == '__main__':
    irs = InformationRetrievalSystem()
    irs.main_menu()
    exit(0)
