from helper import Helper
from flask import Flask, request, jsonify
import consts

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
# helper
helper = Helper()
TITLE_INDEX = helper.get_pickle('/home/igalfernand/postings_gcp/title/index_title.pkl')
TEXT_INDEX = helper.get_pickle('/home/igalfernand/postings_gcp/text/index_text.pkl')
ANCHOR_INDEX = helper.get_pickle('/home/igalfernand/postings_gcp/anchor/index_anchor.pkl')


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query_tokens = helper.get_tokens(query)
    synonyms_dict = consts.synonyms_dict
    for query in query_tokens:
        key = list(filter(lambda x: synonyms_dict[x] == query, synonyms_dict))
        if key:
            query_tokens.append(key[0])
    docs_ids_scores = helper.frequency_ranking(query_tokens, inverted_index=TITLE_INDEX, folder='/home/igalfernand/postings_gcp/title/')
    docs_ids_scores_pagerank = [(doc_id, score, helper.get_page_rank_by_id(id)) for doc_id, score in docs_ids_scores]
    ranking_results_sorted_by_score_then_pr = sorted(docs_ids_scores_pagerank, key=lambda x: (x[1], x[2]), reverse=True)[:100]
    res = helper.get_doc_title_pairs_from_items(ranking_results_sorted_by_score_then_pr)
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query_tokens = helper.get_tokens(query)
    index_terms = set(TEXT_INDEX.df.keys())
    distinct_tokens = set(query_tokens)
    query_tokens_that_exists_in_index = [term for term in distinct_tokens if
                                         term in index_terms]
    if len(query_tokens_that_exists_in_index) == 0:
        return jsonify([])
    Q = helper.generate_query_tfidf_vector(original_query_to_search=query_tokens,
                                           processed_query_to_search=query_tokens_that_exists_in_index,
                                           inverted_index=TEXT_INDEX)
    words, pls = helper.get_posting_iter(TEXT_INDEX, '/home/igalfernand/postings_gcp/text/',
                                         query_tokens=query_tokens_that_exists_in_index)
    D = helper.generate_document_tfidf_matrix(query_tokens_that_exists_in_index, TEXT_INDEX, words, pls)
    sim_dict = helper.cosine_similarity(D, Q)
    top_n_id_score = helper.get_top_n(sim_dict)
    res = helper.get_doc_title_pairs_from_items(top_n_id_score)
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query_tokens = helper.get_tokens(query)
    docs_ids_scores = helper.frequency_ranking(query_tokens, inverted_index=TITLE_INDEX, folder='/home/igalfernand/postings_gcp/title/')
    sorted_ranking_results_docs_ids = sorted(docs_ids_scores, key=lambda item: item[1], reverse=True)
    res = helper.get_doc_title_pairs_from_items(sorted_ranking_results_docs_ids)
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query_tokens = helper.get_tokens(query)
    docs_ids_scores = helper.frequency_ranking(query_tokens, inverted_index=ANCHOR_INDEX, folder='/home/igalfernand/postings_gcp/anchor/')
    sorted_ranking_results_docs_ids = sorted(docs_ids_scores, key=lambda item: item[1], reverse=True)
    res = helper.get_doc_title_pairs_from_items(sorted_ranking_results_docs_ids)
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    else:
        res = [helper.get_page_rank_by_id(doc_id) for doc_id in wiki_ids]
        return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    else:
        res = [helper.get_page_view_by_id(doc_id) for doc_id in wiki_ids]
        return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080)