# Wikipedia-Search-Engine

A basic information retrieval engine on the etire Wikipedia corpus

## Usage

```
SERVER_DOMAIN: http://130.211.215.160:8080

To issue a search query navigate to: SERVER_DOMAIN/search?query=hello+world.
To issue a search query on the body of wikipedia articles navigate to:  SERVER_DOMAIN/search_body?query=hello+world
To issue a search query on the title of wikipedia articles navigate to:  SERVER_DOMAIN/search_title?query=hello+world
To issue a search query on the anchor text of wikipedia articles navigate to:  SERVER_DOMAIN/search_anchor?query=hello+world
To get page rank scores with a json payload of the list of article ids. In python do:
  import requests
  requests.post('SERVER_DOMAIN/get_pagerank', json=[1,5,8])
To get page view scores with a json payload of the list of article ids. In python do:
  import requests
  requests.post('SERVER_DOMAIN/get_pageview', json=[1,5,8]) 
```

## Main code components
**search_frontend.py:** a Flask app that runs on vm server and provides the query functionality for the search engine.

**helper.py:** a library with assisting functions to the search_frontend.py.

**consts.py:** a library with global variables that we are using in the main functions.

**inverted_index_gcp.py:** a library the implements functions for maintaining an inverted index for each index in the project.
