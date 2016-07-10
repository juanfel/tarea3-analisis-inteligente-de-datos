import re, time
from nltk.corpus import stopwords, wordnet
from nltk import WordNetLemmatizer, word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import urllib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#pregunta a)
# train_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.train"
# test_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.dev"
# train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
# test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
ftr = open("train_data.csv", "r")
fts = open("test_data.csv", "r")
rows = [line.split(" ",1) for line in ftr.readlines()]
train_df = pd.DataFrame(rows, columns=['Sentiment','Text'])
train_df['Sentiment'] = pd.to_numeric(train_df['Sentiment'])
rows = [line.split(" ",1) for line in fts.readlines()]
test_df = pd.DataFrame(rows, columns=['Sentiment','Text'])
test_df['Sentiment'] = pd.to_numeric(test_df['Sentiment'])
train_df.shape
test_df.shape
train_positives = train_df.ix[train_df["Sentiment"] == 1]
train_negatives_num = train_df.shape[0] - train_positives.shape[0]
test_positives = test_df.ix[test_df["Sentiment"] == -1]
test_negatives_num = test_df.shape[0] - test_positives.shape[0]

print train_positives.shape[0], train_negatives_num
print test_positives.shape[0], test_negatives_num

##Pregunta b
def word_extractor(text, debug = False):
    wordstemmizer = PorterStemmer()
    commonwords = stopwords.words('english')
    text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
    if debug:
        print text
    words = ""
    wordtokens = [ wordstemmizer.stem(word.lower()) \
                   for word in word_tokenize(text.decode('utf-8', 'ignore')) ]
    for word in wordtokens:
        if word not in commonwords:
            words+=" "+word
    return words

print word_extractor("I love to eat cake")
print word_extractor("I love eating cake")
print word_extractor("I loved eating the cake")
print word_extractor("I do not love eating cake")
print word_extractor("I don't love eating cake") #En resultados aparece n't
print word_extractor("I like drinking soda")
print word_extractor("I dislike to eat meat")
print word_extractor("I disliked to eat meat")
print word_extractor("I absolutely hate to eat cake")
print word_extractor("I never liked to use stemming")

##Pregunta c
def get_wordnet_pos(treebank_tag):
    #Tags entregados por pos_tag no son compatibles con wordnet, hay que
    #convertirlos.
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #Es el default para wordnet tambien
## Es necesario saber que a parte de la oracion corresponde cada palabra
## por eso se usa el pos_tag
def word_extractor2(text, debug = False):
    wordlemmatizer = WordNetLemmatizer()
    commonwords = stopwords.words('english')
    text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
    words = ""
    tagged_words = pos_tag(word_tokenize(text.decode('utf-8', 'ignore')))
    if debug:
        print text
        print tagged_words
    wordtokens = [ wordlemmatizer.lemmatize(word.lower(),pos = get_wordnet_pos(tag)) \
                   for word,tag in tagged_words ]
    for word in wordtokens:
        if word not in commonwords:
            words+=" "+word
    return words


print word_extractor2("I love to eat cake")
print word_extractor2("I love eating cake")
print word_extractor2("I loved eating the cake")
print word_extractor2("I do not love eating cake")
print word_extractor2("I don't love eating cake") #En resultados aparece n't
print word_extractor2("I like drinking soda")
print word_extractor2("I dislike to eat meat")
print word_extractor2("I disliked to eat meat")
print word_extractor2("I absolutely hate to eat cake")
print word_extractor2("I never liked to use stemming")

## Pregunta 3
texts_train = [word_extractor2(text) for text in train_df.Text]
texts_test = [word_extractor2(text) for text in test_df.Text]
vectorizer = CountVectorizer(ngram_range=(1, 1), binary='False')
vectorizer.fit(np.asarray(texts_train))
features_train = vectorizer.transform(texts_train)
features_test = vectorizer.transform(texts_test)
labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)
vocab = vectorizer.get_feature_names()
dist=list(np.array(features_train.sum(axis=0)).reshape(-1,))
for tag, count in sorted(zip(vocab, dist),key = lambda k: k[1]):
    print count, tag
