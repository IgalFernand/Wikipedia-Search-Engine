import math
import pickle
import numpy as np
import pandas as pd
from consts import *
from collections import Counter


class Helper:
    """
    A class that contains helper functions for search_frontend.py interface.
    """

    def __init__(self):
        """
        initialize variables and import pickle files.
        """
        self.PAGERANK = self.get_pickle('/home/igalfernand/postings_gcp/other/pr.pkl')
        self.PAGEVIEWS = self.get_pickle('/home/igalfernand/postings_gcp/other/pageviews-202108-user.pkl')
        self.TITLES = dict(self.get_pickle('/home/igalfernand/postings_gcp/other/doctitles.pkl'))

    def get_pickle(self, pickle_name):
        """
        helper function that imports pickle a file.
        :param pickle_name: .pkl file.
        :return: loaded pickle file.
        """
        with open(pickle_name, 'rb') as file:
            return pickle.load(file)

    def get_tokens(self, text):
        """
        simple tokenization based on assignment 3.
        :param text: a given text string.
        :return: a list of processed tokens.
        """
        list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                          token.group() not in ALL_STOPWORDS]
        return list_of_tokens

    def get_page_rank_by_id(self, doc_id):
        """
        assisting function for get_pagerank in search_frontend.py
        :param doc_id: a given wikipedia document id
        :return: a list of page rank scores the fits the given id.
        """
        try:
            return float(self.PAGERANK.get(doc_id, 0))
        except Exception as ex:
            raise ex

    def get_page_view_by_id(self, doc_id):
        """
         assisting function for get_pageview in search_frontend.py
         :param doc_id: a given wikipedia document id
         :return: a list of page view scores the fits the given id.
         """
        try:
            return int(self.PAGEVIEWS.get(doc_id, 0))
        except Exception as ex:
            raise ex

    def frequency_ranking(self, tokens, inverted_index, folder):
        """
        performs binary retrieval on a given token list and index.
        :param tokens: list of tokens.
        :param inverted_index: .pkl inverted index.
        :param folder: origin folder.
        :return: a list of ranked ids.
        """
        index_terms = set(inverted_index.df.keys())
        # how many terms from the query, exists in each doc?
        terms_in_doc = {}
        distinct_tokens = set(tokens)
        # take only terms that in the index
        query_tokens_that_exists_in_index = [term for term in distinct_tokens if
                                             term in index_terms]
        if len(query_tokens_that_exists_in_index) == 0:
            return []
        words, pls = self.get_posting_iter(inverted_index, folder, query_tokens_that_exists_in_index)
        for term in query_tokens_that_exists_in_index:
            try:
                term_posting_list = pls[words.index(term)]
                for doc_id, freq in term_posting_list:
                    if doc_id != 0:
                        terms_in_doc[doc_id] = terms_in_doc.get(doc_id, 0) + 1
            except Exception as ex:
                raise ex
        return list(terms_in_doc.items())

    def generate_query_tfidf_vector(self, original_query_to_search, processed_query_to_search, inverted_index):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """

        epsilon = .0000001
        Q = {}
        counter = Counter(original_query_to_search)
        for token in processed_query_to_search:
            # term frequency divided by the length of the query
            tf = counter[token] / len(original_query_to_search)
            df = inverted_index.df[token]
            # smoothing
            idf = math.log((len(inverted_index.DL)) / (df + epsilon), 10)
            Q[token] = tf * idf
        return Q

    def get_posting_iter(self, inverted_index, folder, query_tokens):
        """
        This function returning the iterator working with posting list.

        Parameters:
        ----------
        index: inverted index
        """
        words, pls = zip(*inverted_index.posting_lists_iter(folder, query_tokens))
        return words, pls

    def get_candidate_documents_and_scores(self, query_to_search, inverted_index, words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        epsilon = .0000001
        candidates = {}
        for term in np.unique(query_to_search):
            if term in words:
                list_of_doc = pls[words.index(term)]
                normalized_tfidf = [(doc_id, (freq / inverted_index.DL[doc_id]) * math.log(len(inverted_index.DL) / (inverted_index.df[term] + epsilon), 10)) for
                                   doc_id, freq in list_of_doc if doc_id != 0]

                for doc_id, tfidf in normalized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def generate_document_tfidf_matrix(self, query_to_search, inverted_index, words, pls):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.


        words,pls: iterator for working with posting.

        Returns:
        -----------
        DataFrame of tfidf scores.
        """

        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, inverted_index, words, pls)
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), len(query_to_search)))
        D = pd.DataFrame(D)
        D.index = unique_candidates
        D.columns = query_to_search

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf

        return D

    def cosine_similarity(self, D, Q):
        """
        Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
        Generate a dictionary of cosine similarity scores
        key: doc_id
        value: cosine similarity score

        Parameters:
        -----------
        D: DataFrame of tfidf scores.

        Q: vectorized query with tfidf scores

        Returns:
        -----------
        dictionary of cosine similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: cosine similarty score.
        """
        ans = {}
        columns = D.columns
        Q_as_list = []
        for term in columns:
            Q_as_list.append(Q[term])

        for index in D.index:
            row = list(D.loc[index])
            ans[index] = np.dot(row, Q_as_list) / (np.linalg.norm(row) * np.linalg.norm(Q_as_list))

        return ans

    def get_top_n(self, sim_dict, N=100):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """

        return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]

    def get_doc_title_pairs_from_items(self, res):
        """
        maps doc ids to wiki titles.
        :param res: list of doc ids given with scores.
        :return: list of tuples, (wiki id, title).
        """
        return [(item[0], self.TITLES.get(item[0], 0)) for item in res]

    def get_doc_title_pairs_from_id(self, res):
        """
        maps doc ids to wiki titles.
        :param res: list of doc ids.
        :return: list of tuples, (wiki id, title).
        """
        return [(id, self.TITLES.get(id, 0)) for id in res]