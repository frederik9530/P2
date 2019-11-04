# OPEN DATA SCIENCE, PORTFOLIO_2 (DRAFT)
# I have just sent my paper in it's so-far form, in order to get feedback - so basically i
# have almost gone through it all until 4 (the topic modelling + graphics)


# codename: Hong Kong (or hong)


# 1. The collection

# So this first long body, is (obviously) a copy-paste from the guardian
# that I have inserted in order to get the data from the API

import json
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

# This creates two subdirectories called "theguardian" and "collection"
ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)

# Sample URL
#
# http://content.guardianapis.com/search?from-date=2016-01-02&
# to-date=2016-01-02&order-by=newest&show-fields=all&page-size=200
# &api-key=your-api-key-goes-here

# Change this for your API key:
MY_API_KEY = '3aca8537-1937-4bcd-b313-b1371548bb69'

API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "", # leave empty, change start_date / end_date variables instead
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY
}

# day iteration from here:
# http://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates

# Update these dates to suit your own needs.
start_date = date(2019, 7, 28)
end_date = date(2019,10, 28)

dayrange = range((end_date - start_date).days + 1)
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    fname = join(ARTICLES_DIR, datestr + '.json')
    if not exists(fname):
        # then let's download it
        print("Downloading", datestr)
        all_results = []
        my_params['from-date'] = datestr
        my_params['to-date'] = datestr
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            print("...page", current_page)
            my_params['page'] = current_page
            resp = requests.get(API_ENDPOINT, my_params)
            data = resp.json()
            all_results.extend(data['response']['results'])
            # if there is more than one page
            current_page += 1
            total_pages = data['response']['pages']

        with open(fname, 'w') as f:
            print("Writing to", fname)

            # re-serialize it for pretty indentation
            f.write(json.dumps(all_results, indent=2))

import json
import os

# Update to the directory that contains your json files
# Note the trailing /
directory_name = "theguardian/collection/"

ids = list()
texts = list()
sections = list()
for filename in os.listdir(directory_name):
    if filename.endswith(".json"):
        with open(directory_name + filename) as json_file:
            data = json.load(json_file)
            for article in data:
                id = article['id']
                fields = article['fields']
                text = fields['bodyText'] if fields['bodyText'] else ""
                ids.append(id)
                texts.append(text)
                section = article['sectionId']	# Id name each article gets by The Guardian
                sections.append(section) # Adding each item to a list as above "sections = list()"


print("Number of ids: %d" % len(ids))
print("Number of texts: %d" % len(texts))

# The set of articles-variable is called: texts

# 2.  Pre-process and describe your collection

#Research question: With with topics was the 2019 Hong Kong demonstrations discussed

# First off, i'm creating an environment that seems relevant, according to
# the previous text-mining exercises
import sklearn
import numpy as np
import pandas as pd
import nltk
import matplotlib

# And then i'll describe the raw data
len(texts)
# This particular collection from the Guardian contains 20034 texts
# Which I guess, we already knew, but now we know-know



all_lengths = list()
for text in texts:
    all_lengths.append(len(text))
print("Total sum: %i" % sum(all_lengths))
# This means, that these roughly three months of articles alltogether consists of 102811552 characters


# Then i use 'stats' from Scipy to do some more general statistics on the entire collection
from scipy import stats 
stats.describe(all_lengths)

#drop evt nedenstående
# Then via a simple tokenization, we'll find the sum of words
word_count = 0
for text in texts:
    words = text.split()
    word_count = word_count + len(words)
word_count
print('there are %i words in total' % word_count) 

# And the unique words
all_words = list()
for text in texts:
  words = text.split()
  all_words.extend(words)
print("Word count: %i" % len(all_words))
unique_words = set(all_words)
unique_word_count = len(unique_words)
print("Unique word count: %i" % unique_word_count)
# (Which amounts to 507608)


# I will make a document-term matrix via sklearn's CountVectorizer
#My preprocessing consists of setting the maxfeatures for 10.000, add nltk's list of english stopwords
# and via a RegEx, we'll define terms that consists of two or more alphabetic symbols + hyphens.

#But first...
idxs = []
sub_texts = []
for i, section in enumerate(sections):
    if section in ['news','world news', 'politics', 'society', 'world']:
        idxs.append(i)
        sub_texts.append(texts[i])
len(idxs)
# ... I derive the sections 'news',' world news', 'politics, 'society', 'world' and create a list called 'sub_texts'.
# Which gives me a total of just 3933 texts

len(sections)
unique_sections = set(sections)
len(unique_sections)
unique_sections
# I check out the list of unique sections for my data, to choose the relevant sub_texts.
# Just to make sure, that I am happy with my choice of just the above-mentioned 5


# And then we use these in out CountVectorizer:
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as sw
model_vect = CountVectorizer(max_features = 10000, stop_words = sw.words('english'), token_pattern = r'[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vect = model_vect.fit_transform(sub_texts)
data_vect
np.shape(data_vect) 

# The resulting term-document-matrix is a sparse matrix (3933, 10000)
# with  rows=texts, columns: terms, cells: counts

# Then we'll take a look at the top indexes
counts = data_vect.sum(axis = 0).A1
print(counts)
top_idxs = (-counts).argsort()[:10]
print(top_idxs)

# - In order to see the top words they represent:
inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words: %s" % top_words)
# Which are not very revealing for anything - but none the less they are actual words.
# Good enough - but i will also take a random sample of the words,
# this also to make sure, my preprocessing was succesful


import random
random.sample(unique_words, 20)




# 3. Select articles using a query
# I want to find articles related to the demonstrations in Hong Kong
#So i wanted originally to just make my query 'Hong Kong' but I landed on:
# 'hong', 'protests' and 'extradition'

terms = ['hong', 'protests', 'extradition']
# After several tries with versatile letters and "hong kong", i just settled for
# 'hong'. It's not like 'hong' og 'kong' is seperate words in english anyway

term_idxs = [model_vect.vocabulary_.get(term) for term in terms]
term_counts = [counts[idx] for idx in term_idxs]
print(term_counts)

#Here we get the term counts for each of the three words in our query

from sklearn.feature_extraction.text import TfidfTransformer
model_tfidf = TfidfTransformer()
data_tfidf = model_tfidf.fit_transform(data_vect)
data_tfidf
# And then here I transform the count to a tfidf representation


idfs = model_tfidf.idf_
term_idfs = [idfs[idx] for idx in term_idxs]
term_idfs
# And I get the individual weights for my selected terms

df = pd.DataFrame(columns=['count', 'idf'], index=terms, data=zip(term_counts,term_idfs))
df
# Here i use pandas to make a dataframe, that will let me compare my term counts with
# my term weights. The one that have the largest difference is 'extradition', which i 
# guess means that it has a higher semantic value compared to how often it is represented
# in my texts (?)

query = " ".join(terms)
query

# And then let's turn the query into a Query Vector
from sklearn.feature_extraction.text import TfidfTransformer
model_tfidf = TfidfTransformer()
query_vect_counts = model_vect.transform([query])
query_vect = model_tfidf.fit_transform(query_vect_counts)
query_vect


from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(query_vect, data_tfidf)
sims
np.shape(sims)


sims_sorted_idx = (-sims).argsort()
sims_sorted_idx
np.shape(sims_sorted_idx)

# Now - the above lines of code are hardfor me to explain, but they are related
# to investigating the cosine-similarity of my terms (/query?)

# Which should lead the below text (first in the matrix), to be the most relevant 
# text for my query
sub_texts[sims_sorted_idx[0,0]]
# The article is an extremely relevant match

print("Shape of 2-D array sims: (%i, %i)" % (len(sims), len(sims[0,:])) )
df = pd.DataFrame(data=zip(sims_sorted_idx[0,:], sims[0,sims_sorted_idx[0,:]]), columns=["index", "cosine_similarity"])
df[0:10]
# Here we have the (allegedly) top-ten texts for my query

# And this is about as far, i still (almost) got my head in this



# Opgave 4 

#LDA-topic modelling
from sklearn.decomposition import LatentDirichletAllocation
model_lda = LatentDirichletAllocation(n_components=4, random_state=0)
data_lda = model_lda.fit_transform(data_vect)

import numpy as np

for i, term_weights in enumerate(model_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = ["%s (%.3f)" % (model_vect.get_feature_names()[idx], term_weights[idx]) for idx in top_idxs]
    print("Topic %d: %s" % (i, ", ".join(top_words)))
    
# Neither of the resulting topics are relevant, to my research question. Maybe
# The query relates to too wide a variety of articles - since a lot of articles
# with relation to protests (also outside of China) and politics appear, as well
# as articles regarding Hong Kong in general.











