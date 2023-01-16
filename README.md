# Wikipedia-Search-Engine

A simple and efficient information retrieval engine that allows users to search the entire Wikipedia corpus. This engine includes four different types of search options: searching the body of articles, searching the title of articles, searching the anchor text of articles and a combined search option. It also has two additional options to retrieve page rank and page view scores of articles.

## Usage

```
To issue a search query navigate to: [Hidden URL]/search?query=hello+world.
To issue a search query on the body of Wikipedia articles navigate to: [Hidden URL]/search_body?query=hello+world
To issue a search query on the title of Wikipedia articles navigate to: [Hidden URL]/search_title?query=hello+world
To issue a search query on the anchor text of Wikipedia articles navigate to: [Hidden URL]/search_anchor?query=hello+world
To get page rank scores with a json payload of the list of article ids. In python do:
  import requests
  requests.post([Hidden URL]/get_pagerank', json=[1,5,8])
To get page view scores with a json payload of the list of article ids. In python do:
  import requests
  requests.post([Hidden URL]/get_pageview', json=[1,5,8]) 
```

## Main code components

* `search_frontend.py`: A Flask app that runs on the server and provides the query functionality for the search engine.
* `helper.py`: A library with assisting functions to search_frontend.py.
* `consts.py`: A library with global variables that we are using in the main functions.
* `inverted_index_gcp.py`: A library that implements functions for maintaining an inverted index for each index in the project.
