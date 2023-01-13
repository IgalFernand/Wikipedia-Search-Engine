import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1

nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)

# create a words - synonyms dictionary based on assignment 2 top n corpus stopwords
top_title_words = ["district","house","season","football","amara","disambiguation","station","2008","school","list","team","ban","carolina","john","boston","east","pirates","celtics","album","historic","council","county","new","island","light","league","united","college","national","farm","film","community","toyota","corolla","george","2007","louis","men's","footballer","baseball","lee","mill","william","championship","states","david","university","election","american","sports","hill","club","site","park","blues","international","baltimore","european","charles","series","cup","henry","wrestling","summer","olympics","trox","company","church","thomas","martin","arnold","south","james","high","khan","musician","1992","york","maryland","williams","jim","bill","street","war","open","north","surname","metro","railway","group","rhode","regiment","black","1920","greco-roman","rules","discography","paraguay","destinations","people"]
synonyms = ["region","home","period","soccer","","","stop","","college","","group","prohibit","","","","","Caribbean","","music","historical","board","state","latest","isle","","leagues","usa","university","","","movie","public","","","","","","men","player","","","","","","county","","college","vote","","sport","hill","","location","garden","","global","","europa","","","world","","tussle","","olympic","","firm","","","","","","","top","","music","","new","","","","","road","battle","","","name","subway","rail","team","","unit","","","","rule","","Asuncin","destination","nation"]

synonyms_dict = dict(zip(top_title_words, synonyms))