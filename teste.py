import lda
from lda.utils import lists_to_matrix
import lda.datasets
import re
from random import shuffle
from statistics import mean

import matplotlib.pyplot as plt

import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm
import textmining
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


#gensim text filtering
# Define functions for stopwords, bigrams, trigrams and lemmatization


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        item = row
        row = sorted(item, key=lambda x: x[1], reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

#[END]defined functions
#
#



data = {'rating': [], 'text': []}
flag = 0

#amazon musical instruments database
with open('Musical_Instruments.json') as f:
    for line in tqdm(f):

        #flag to limit data lines to 1000
        #remove in final version, it's just to test
        if flag > 1000:
            break

        review = json.loads(line)

        #filter data from dataset dict
        check = review.get('reviewText')
        if check:
            data['text'].append(review['reviewText'])

        check = review.get('overall')
        if check:
            data['rating'].append(review['overall'])
        
        flag+=1
        
#df = pd.DataFrame(data)
#df.to_csv('musical_review.csv', index= True)


words_list = [] #vocab
words_by_review = []
reviews_list = [] #reviews
tdm = textmining.TermDocumentMatrix() #tdm


sia = SentimentIntensityAnalyzer()
#print(sia.polarity_scores("Wow, NLTK is really powerful!"))

for review in data['text']:
    separated_words_review = []
    reviews_list.append(review)
    normalized_review = re.sub('''[.,:?!'"()]''',"", review) #review without special characters
    split_words = normalized_review.split()
    for word in split_words:
        if word not in words_list:
            words_list.append(word)
        separated_words_review.append(word)
    words_by_review.append(separated_words_review)
    tdm.add_doc(normalized_review)


df = pd.DataFrame(data=data)

df['scores'] = df['text'].apply(lambda review : sia.polarity_scores(review))
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound']) 

print("-------------------------------------------")
print("Dataframe")
print(df.head()) 
print("-------------------------------------------")

review = tuple(reviews_list) #final review's info

review_words = set(words_list)
stop_words = set(stopwords.words('english')) #set of words that need to be cleaned
filtered_words = []


for w in review_words:
    if w not in stop_words:
        filtered_words.append(w)
    
vocab = tuple(filtered_words) #final vocabulary's info

#---------------------------------------------------------------------

# Build the bigram and trigram models
bigram = gensim.models.Phrases(words_by_review, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[words_by_review], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

review_nostops = remove_stopwords(words_by_review)
review_bigrams = make_bigrams(review_nostops)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

review_lemmatized = lemmatization(review_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(review_lemmatized)

# Create Corpus
texts = review_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

#LDA topics defined
print("-------------------------------------------")
print("LDA Topics")
pprint(lda_model.print_topics())
print("-------------------------------------------")


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

df['Topic'] = df_topic_sents_keywords

print("-------------------------------------------")
print("Complete dataframe")
print(df.head())
print("-------------------------------------------")