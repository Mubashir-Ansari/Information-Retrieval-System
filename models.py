# Contains all retrieval models.

import porter
import hashlib
import random
import math
from document import Document
from collections import defaultdict, Counter
from abc import ABC, abstractmethod

class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Converts a document into its model-specific representation.
        This is an abstract method and not meant to be edited. Implement it in the subclasses!
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A representation of the document. Data type and content depend on the implemented model.
        """
        raise NotImplementedError()

    @abstractmethod
    def query_to_representation(self, query: str):
        """
        Determines the representation of a query according to the model's concept.
        :param query: Search query of the user
        :return: Query representation in whatever data type or format is required by the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation according to the model's concept.
        :param document_representation: Data that describes one document
        :param query_representation:  Data that describes a query
        :return: Numerical approximation of the similarity between the query and document representation. Higher is
        "more relevant", lower is "less relevant".
        """
        raise NotImplementedError()


class LinearBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods and __init__() in this class. (PR02)
    def __init__(self):
        self.documents = []
        self.stemming=False

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        doc_id = document.document_id
        self.stemming = stemming

        if stemming and stopword_filtering:
            terms = document.stemmed_filtered_terms
        elif stopword_filtering:
            terms = document.filtered_terms
        elif stemming:
            terms = document.stemmed_terms
        else:
            terms=document.terms

        if not any(doc_id==doc[0] for doc in self.documents):    
            self.documents.append((document.document_id, terms))
        return document.document_id

    def query_to_representation(self, query: str):
        return query.lower()

    def match(self, document_representation, query_representation) -> float:
        matching_docs = self.evaluate_query(query_representation)
        return 1.0 if document_representation in matching_docs else 0.0
    
    def evaluate_query(self, query_representation: str):
        tokens = self.tokenize_query(query_representation)
        if self.stemming:
            tokens=[porter.stem_term(token) for token in tokens]

        return self.evaluate_tokens(tokens)

    def tokenize_query(self, query: str):
        tokens = []
        current_term = ''
        for char in query:
            if char in '&|()-':
                if current_term:
                    tokens.append(current_term)
                    current_term = ''
                tokens.append(char)
            else:
                current_term += char
        if current_term:
            tokens.append(current_term)
        return tokens

    def evaluate_tokens(self, tokens):
        operation_stack = []
        operator_stack = []
        operator_precedence = {'|': 1, '&': 2, '-': 3}

        def apply_operation():
            operation = operator_stack.pop()
            if operation == '-':
                term = operation_stack.pop()
                result = set(doc_id for doc_id, terms in self.documents) - term
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
                term_docs = set(doc_id for doc_id, terms in self.documents if token in terms)
                operation_stack.append(term_docs)
            i += 1

        while operator_stack:
            apply_operation()

        return operation_stack[0] if operation_stack else set()
    
    def __str__(self):
        return 'Boolean Model (Linear)'


class InvertedListBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods and __init__() in this class. (PR03)
    def __init__(self):
        self.inverted_index = {}
        self.doc_ids = set()
        self.stemming=False
        

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        terms = document.terms
        self.stemming = stemming
        

        if stopword_filtering and stemming:
            terms = document.stemmed_filtered_terms
        elif stemming:
            terms = document.stemmed_terms
        elif stopword_filtering:
            terms = document.filtered_terms
            
        # Build the inverted index
        for term in terms:
            if term not in self.inverted_index:
                self.inverted_index[term] = set()
            self.inverted_index[term].add(document.document_id)
        
        self.doc_ids.add(document.document_id)
        return document.document_id

    def query_to_representation(self, query: str):
        terms = query.lower()
        return terms
        

    def match(self, document_representation, query_representation) -> float:
        matching_doc_ids = self.evaluate_query(query_representation)
        return 1.0 if document_representation in matching_doc_ids else 0.0

    def evaluate_query(self, query_representation):
        tokens = self.tokenize_query(query_representation)
        if self.stemming:
            tokens=[porter.stem_term(token) for token in tokens]

        return self.evaluate_tokens(tokens)
    
    def tokenize_query(self, query):
        tokens = []
        current_term = ''
        for char in query:
            if char in '&|()-':
                if current_term:
                    tokens.append(current_term)
                    current_term = ''
                tokens.append(char)
            else:
                current_term += char
        if current_term:
            tokens.append(current_term)
        return tokens

    def evaluate_tokens(self, tokens):
        operation_stack = []
        operator_stack = []
        operator_precedence = {'|': 1, '&': 2, '-': 3}

        def apply_operation():
            operation = operator_stack.pop()
            if operation == '-':
                term = operation_stack.pop()
                result = self.doc_ids - term
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
                if token in self.inverted_index:
                    operation_stack.append(self.inverted_index[token])
                else:
                    operation_stack.append(set())
            i += 1

        while operator_stack:
            apply_operation()

        return operation_stack[0] if operation_stack else set()

    def __str__(self):
        return 'Boolean Model (Inverted List)'

class SignatureBasedBooleanModel(RetrievalModel):
    def __init__(self, F=64, D=4, m=20):
        self.F = F
        self.D = D
        self.m = m
        self.term_signatures = {}
        self.block_signatures = {}
        self.block_terms = {}
        self.current_block_id = 1
        self.documents = []
        self.stemming=False
        self.stopWords=False

    def _hash_term(self, term: str) -> str:
        hash_obj = hashlib.md5(term.encode())
        hex_hash = hash_obj.hexdigest()
        binary_hash = bin(int(hex_hash, 16))[2:].zfill(self.F)

        if len(binary_hash) > self.F:
            binary_hash = binary_hash[:self.F]
        elif len(binary_hash) < self.F:
            binary_hash = binary_hash.zfill(self.F)

        random.seed(int(hex_hash, 16))
        positions_to_set = random.sample(range(self.F), self.m)
        modified_hash_list = ['0'] * self.F

        for pos in positions_to_set:
            modified_hash_list[pos] = '1'

        modified_hash = ''.join(modified_hash_list)
        return modified_hash
    
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        doc_id = document.document_id
        terms = document.terms
        
        self.stemming = stemming
        self.stopWords=stopword_filtering

        if not any(doc.document_id == doc_id for doc in self.documents):
            self.documents.append(document)
                
        if stopword_filtering:
            terms = document.filtered_terms
        elif stemming:
            terms = document.stemmed_terms
        elif stemming and stopword_filtering:
            terms = document.stemmed_filtered_terms

        for term in terms:
            if term not in self.term_signatures:
                self.term_signatures[term] = self._hash_term(term)
        
        num_terms = len(terms)
        
        for i in range(0, num_terms, self.D):
            block_id = self.current_block_id
            block_terms = terms[i:i + self.D]
            block_signature = '0' * self.F
            
            # Compute block signature using bitwise OR across all terms in the block
            for term in block_terms:
                term_hash = self.term_signatures[term]
                block_signature = self._bitwise_or(block_signature, term_hash)
            
            self.block_signatures[block_id] = block_signature
            self.block_terms[block_id] = (doc_id, block_terms)  # Store (document ID, terms)
            self.current_block_id += 1
        return document.document_id
    
    def _bitwise_or(self, hash1: str, hash2: str) -> str:
        result = ['1' if bit1 == '1' or bit2 == '1' else '0' for bit1, bit2 in zip(hash1, hash2)]
        return ''.join(result)
    
    def _bitwise_and(self, hash1: str, hash2: str) -> str:
        result = ['1' if bit1 == '1' and bit2 == '1' else '0' for bit1, bit2 in zip(hash1, hash2)]
        return ''.join(result)
    
    def query_to_representation(self, query: str):
        return query.lower()
    
    def match(self, document_representation, query_representation) -> float:
        query_signature_or_docs = self.evaluate_query(query_representation)
        # Determine if the query is a single term
        is_single_term = len(query_representation.split()) == 1 and '&' not in query_representation and '|' not in query_representation
        
        if isinstance(query_signature_or_docs, str):
            query_signature = query_signature_or_docs
            matched_docs = set()
            
            # Perform bitwise AND for conjunctions
            for block_id, block_signature in self.block_signatures.items():
                if self._bitwise_and(query_signature, block_signature) == query_signature:
                    doc_id, _ = self.block_terms[block_id]
                    matched_docs.add(doc_id)

            if is_single_term:
                resulting_docs = self.linear_search(matched_docs, query_representation.split())
            else:
                resulting_docs = matched_docs
        else:
            resulting_docs = query_signature_or_docs
            
        return 1.0 if document_representation in resulting_docs else 0.0
    
    def evaluate_query(self, query: str):
        tokens = self.tokenize_query(query)
        if self.stemming:
            tokens=[porter.stem_term(token) for token in tokens]

        return self.evaluate_tokens(tokens)

    def tokenize_query(self, query: str):
        tokens = []
        current_term = ''
        for char in query:
            if char in '&|()':
                if current_term:
                    tokens.append(current_term)
                    current_term = ''
                tokens.append(char)
            else:
                current_term += char
        if current_term:
            tokens.append(current_term)
        return tokens

    def evaluate_tokens(self, tokens):
        operand_stack = []
        operator_stack = []
        operator_precedence = {'|': 1, '&': 2}

        def apply_operation():
            operator = operator_stack.pop()
            right_operand = operand_stack.pop()
            left_operand = operand_stack.pop()
            if operator == '&':
                result = self._bitwise_or(left_operand, right_operand)
            elif operator == '|':
                result = self._union_documents(left_operand, right_operand)
            operand_stack.append(result)

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    apply_operation()
                operator_stack.pop()  # Remove the '('
            elif token in operator_precedence:
                while (operator_stack and operator_stack[-1] in operator_precedence and
                       operator_precedence[token] <= operator_precedence[operator_stack[-1]]):
                    apply_operation()
                operator_stack.append(token)
            else:  # Operand
                if token in self.term_signatures:
                    operand_stack.append(self.term_signatures[token])
                else:
                    operand_stack.append('0' * self.F)
            i += 1

        while operator_stack:
            apply_operation()

        return operand_stack[0]

    def _union_documents(self, signature1: any, signature2: any) -> str:
        matched_docs1 = self._get_matched_docs(signature1) if isinstance(signature1, str) else signature1
        matched_docs2 = self._get_matched_docs(signature2) if isinstance(signature2, str) else signature2
        return matched_docs1 | matched_docs2

    def _get_matched_docs(self, signature: str) -> set:
        matched_docs = set()
        for block_id, block_signature in self.block_signatures.items():
            if self._bitwise_and(signature, block_signature) == signature:
                doc_id, _ = self.block_terms[block_id]
                matched_docs.add(doc_id)
        return matched_docs


    def linear_search(self, matched_docs, query_terms) -> set:
        resulting_docs = set()
        for doc_id in matched_docs:
            for document in self.documents:
                if doc_id == document.document_id:
                    if self.stopWords and self.stemming:
                        doc_terms = document.stemmed_filtered_terms
                    elif self.stopWords:
                        doc_terms = document.filtered_terms
                    elif self.stemming:
                        doc_terms = document.stemmed_terms
                    else:
                        doc_terms = document.terms

                    if all(term in doc_terms for term in query_terms):
                        resulting_docs.add(document.document_id)
        return resulting_docs
    

    def __str__(self):
        return 'Boolean Model (Signatures)'

class VectorSpaceModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.document_freq = defaultdict(int)
        self.doc_length = {}
        self.N = 0
        self.stemming=False

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        self.stemming=stemming

        if stopword_filtering and stemming:
            terms = document.stemmed_filtered_terms
        elif stopword_filtering:
            terms = document.filtered_terms
        elif stemming:
            terms = document.stemmed_terms
        else:
            terms = document.terms

        term_counts = Counter(terms)
        doc_id = document.document_id

        for term, count in term_counts.items():
            self.inverted_index[term].append((doc_id, count))
            self.document_freq[term] += 1

        self.N += 1
        tfidf_representation = {}
        for term, count in term_counts.items():
            tf = count / len(terms)
            idf = math.log((self.N) / (1 + self.document_freq[term]))
            tfidf_representation[term] = tf * idf

        norm_denominator = math.sqrt(sum((tfidf_representation[term])**2 for term in term_counts))

        for term in tfidf_representation:
            tfidf_representation[term] /= norm_denominator

        self.doc_length[doc_id] = norm_denominator
        
        return tfidf_representation

    def query_to_representation(self, query: str):
        query_terms = query.split()
        if self.stemming:
            query_terms=[porter.stem_term(token) for token in query_terms]
        term_counts = Counter(query_terms)
        max_term_count = max(term_counts.values(), default=1)
        tfidf_representation = {}

        for term, count in term_counts.items():
            tf = count
            max_tf = max_term_count
            if max_tf > 0:
                normalized_tf = 0.5 + 0.5 * (tf / max_tf)
            else:
                normalized_tf = 0
            idf = math.log((self.N) / (1 + self.document_freq.get(term, 0)))
            tfidf_representation[term] = normalized_tf * idf

        return tfidf_representation

    def match(self, query_representation,document_representation) -> float:
        dot_product = sum(query_representation.get(term, 0) * document_representation.get(term, 0) for term in set(query_representation.keys()).union(set(document_representation.keys())))
        magnitude1 = math.sqrt(sum(val ** 2 for val in query_representation.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in document_representation.values()))
        epsilon = 1e-10
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return abs(dot_product / (magnitude1 * magnitude2 + epsilon))
    
    def __str__(self):
        return 'Vector Space Model'


class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Fuzzy Set Model'
