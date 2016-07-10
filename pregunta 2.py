from sklearn.naive_bayes import BernoulliNB, random
from sklearn.metrics import classification_report
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
def word_extractor(text, useStopwords = True, debug = False):
    wordstemmizer = PorterStemmer()
    if useStopwords:
        commonwords = stopwords.words('english')
    else:
        commonwords = []
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
def word_extractor2(text, useStopwords = True, debug = False):
    wordlemmatizer = WordNetLemmatizer()
    if useStopwords:
        commonwords = stopwords.words('english')
    else:
        commonwords = []
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

## Pregunta d
def generate_features(train_df,test_df,extractor = word_extractor2, useWordstops = True):
    #Permite generar features en base a los datos iniciales, y permite especificar
    #parametros tales como el extractor o si usar wordstops
    texts_train = [extractor(text, useWordstops) for text in train_df.Text]
    texts_test = [extractor(text, useWordstops) for text in test_df.Text]
    vectorizer = CountVectorizer(ngram_range=(1, 1), binary='False')
    vectorizer.fit(np.asarray(texts_train))
    features_train = vectorizer.transform(texts_train)
    features_test = vectorizer.transform(texts_test)
    labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
    labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)
    vocab = vectorizer.get_feature_names()
    return features_train, features_test, labels_train, labels_test, vocab

features_train, features_test, labels_train, labels_test, vocab = generate_features(train_df,test_df)
dist=list(np.array(features_train.sum(axis=0)).reshape(-1,))
for tag, count in sorted(zip(vocab, dist),key = lambda k: k[1]):
    print count, tag

## Pregunta e
def score_the_model(model,x,y,xt,yt,text):
    acc_tr = model.score(x,y)
    acc_test = model.score(xt[:-1],yt[:-1])
    print "Training Accuracy %s: %f"%(text,acc_tr)
    print "Test Accuracy %s: %f"%(text,acc_test)
    print "Detailed Analysis Testing Results ..."
    print(classification_report(yt, model.predict(xt), target_names=['+','-']))

## Pregunta h
def do_NAIVE_BAYES(x,y,xt,yt):
    model = BernoulliNB()
    model = model.fit(x, y)
    score_the_model(model,x,y,xt,yt,"BernoulliNB")
    return model
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)
spl = random.sample(xrange(len(test_pred)), 15)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
    print sentiment, text
