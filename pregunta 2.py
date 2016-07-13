from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import random
from sklearn.metrics import classification_report, f1_score
import re, time
from nltk.corpus import stopwords, wordnet
from nltk import WordNetLemmatizer, word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import urllib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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
    if useStopwords == True:
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
    if useStopwords == True:
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
    print "Features listas"
    return features_train, features_test, labels_train, labels_test, vocab


## Crea features para cada combinacion de extractor-wordstop
## Esto es porque este calculo es de los mas lentos, por lo cual no conviene
## hacerlo cada vez que se quiera correr un test.
we2_features_wordstop = generate_features(train_df,test_df)
we2_features_non_wordstop = generate_features(train_df,test_df,word_extractor2,False)
we_features_wordstop = generate_features(train_df,test_df,word_extractor,True)
we_features_non_wordstop = generate_features(train_df,test_df,word_extractor,False)
def get_features(extractor,useWordstops):
    #Obtiene el vector de features de acuerdo a los parametros de arriba
    #Nada elegante.
    if extractor.__name__ == "word_extractor2":
        if useWordstops:
            return we2_features_wordstop
        else:
            return we2_features_non_wordstop
    else:
        if useWordstops:
            return we_features_wordstop
        else:
            return we_features_non_wordstop

features_train, features_test, labels_train, labels_test, vocab = get_features(word_extractor2,True)
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
    prediction = model.predict(xt)
    print(classification_report(yt, prediction, target_names=['+','-']))
    return f1_score(yt, prediction ), acc_tr, acc_test

## Pregunta f
def do_NAIVE_BAYES(x,y,xt,yt):
    model = BernoulliNB()
    model = model.fit(x, y)
    f1 = score_the_model(model,x,y,xt,yt,"BernoulliNB")
    return model, f1

def test_Model(train_df,test_df,model_function,extract_function = word_extractor2, useStopwords = True, multipleModels = False, useProbabilities = True, usePredictionValue = True):
    #Prueba el modelo usando una muestra aleatoria
    #Si multipleModels = true, asume que la funcion va a entregar un iterable
    #con funciones
    #Se hace asi para que salgan resultados ordenados
    #Retorna el o los valores de f1_score de los modelos
    features_train, features_test, labels_train, labels_test, vocab = get_features(extract_function,useStopwords)
    print "Function name %s"%extract_function.__name__
    print "Use stopwords %s"%useStopwords
    if multipleModels:
        model_functions = model_function()
    else:
        model_functions = [model_function]

    return_values = []

    for mod_fun in model_functions:
        model, scores = mod_fun(features_train,labels_train,features_test,labels_test)
        sample_size = features_test.shape[0]
        spl = random.sample(xrange(sample_size), 15)
        if useProbabilities == True:
            test_prob = model.predict_proba(features_test)
            if not usePredictionValue:
                for text, sentiment in zip(test_df.Text[spl], test_prob[spl]):
                    print sentiment, text
        if usePredictionValue == True:
            test_pred = model.predict(features_test)
            if useProbabilities:
                for text, prob, sentiment in zip(test_df.Text[spl], test_prob[spl], test_pred[spl]):
                   print sentiment, prob, text 
            else:
                for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
                    print sentiment, text
        return_values.append(scores)
    if multipleModels:
        return return_values
    else:
        return return_values[0]
# Casos de prueba
naive_all_values = []
naive_all_values.append(test_Model(train_df,test_df,do_NAIVE_BAYES, word_extractor2, True))
naive_all_values.append(test_Model(train_df,test_df,do_NAIVE_BAYES, word_extractor2, False))
naive_all_values.append(test_Model(train_df,test_df,do_NAIVE_BAYES, word_extractor, True))
naive_all_values.append(test_Model(train_df,test_df,do_NAIVE_BAYES, word_extractor, False))

naive_values,naive_train_acc, naive_test_acc = zip(*naive_all_values)
## Pregunta g
def do_MULTINOMIAL(x,y,xt,yt):
    model = MultinomialNB()
    model = model.fit(x, y)
    f1 = score_the_model(model,x,y,xt,yt,"MULTINOMIAL")
    return model, f1
multinomial_all_values = []
multinomial_all_values.append(test_Model(train_df,test_df,do_MULTINOMIAL, word_extractor2, True))
multinomial_all_values.append(test_Model(train_df,test_df,do_MULTINOMIAL, word_extractor2, False))
multinomial_all_values.append(test_Model(train_df,test_df,do_MULTINOMIAL, word_extractor, True))
multinomial_all_values.append(test_Model(train_df,test_df,do_MULTINOMIAL, word_extractor, False))

multinomial_values,multinomial_train_acc, multinomial_test_acc = zip(*multinomial_all_values)
## Pregunta h
logit_cs = [0.01,0.1,0.9,1,10,100,1000]
def do_LOGITS():
    #Crea varias funciones do_LOGIT para cada valor de c
    #Es necesario hacer esto para no reescribir codigo
    #Usa closures y es una funcion generadora
    start_t = time.time()
    Cs = logit_cs
    for C in Cs:
        def do_LOGIT(x,y,xt,yt):
            print "Usando C= %f"%C
            model = LogisticRegression(penalty='l2',C=C)
            model = model.fit(x, y)
            f1 = score_the_model(model,x,y,xt,yt,"LOGISTIC")
            return model, f1
        yield do_LOGIT
logit_all_values = []
logit_all_values.append(zip(*test_Model(train_df,test_df,do_LOGITS, word_extractor2, True,True)))
logit_all_values.append(zip(*test_Model(train_df,test_df,do_LOGITS, word_extractor2, False,True)))
logit_all_values.append(zip(*test_Model(train_df,test_df,do_LOGITS, word_extractor, True,True)))
logit_all_values.append(zip(*test_Model(train_df,test_df,do_LOGITS, word_extractor, False,True)))

logit_values,logit_train_acc, logit_test_acc = zip(*logit_all_values)

logit_f1_max = max([(max(a), a.index(max(a))) for a in logit_values])
## Pregunta i
svm_cs = [0.01,0.04,0.05,0.06,0.1,0.5,1,10,100,1000]

def do_SVMS():
    #Crea varias funciones do_SVM para cada valor de c
    Cs = svm_cs
    for C in Cs:
        def do_SVM(x,y,xt,yt):
            print "El valor de C que se esta probando: %f"%C
            model = SVC(C=C,kernel='linear',probability=True)
            model = model.fit(x, y)
            f1 = score_the_model(model,x,y,xt,yt,"SVM")
            return model, f1
        yield do_SVM
# #Muy lentos no usar
# test_Model(train_df,test_df,do_SVMS, word_extractor2, True,True,False)
# test_Model(train_df,test_df,do_SVMS, word_extractor2, False,True,False)
# test_Model(train_df,test_df,do_SVMS, word_extractor, True,True,False)
# test_Model(train_df,test_df,do_SVMS, word_extractor, False,True,False)
def do_Linear_SVMS():
    #Crea varias funciones do_SVM para cada valor de c
    Cs = svm_cs
    for C in Cs:
        def do_SVM(x,y,xt,yt):
            print "El valor de C que se esta probando: %f"%C
            model = LinearSVC(C=C)
            model = model.fit(x, y)
            f1 = score_the_model(model,x,y,xt,yt,"SVM")
            return model,f1
        yield do_SVM
svm_all_values = []
svm_all_values.append(zip(*test_Model(train_df,test_df,do_Linear_SVMS, word_extractor2, True,True,False)))
svm_all_values.append(zip(*test_Model(train_df,test_df,do_Linear_SVMS, word_extractor2, False,True,False)))
svm_all_values.append(zip(*test_Model(train_df,test_df,do_Linear_SVMS, word_extractor, True,True,False)))
svm_all_values.append(zip(*test_Model(train_df,test_df,do_Linear_SVMS, word_extractor, False,True,False)))

svm_values,svm_train_acc, svm_test_acc = zip(*svm_all_values)
svm_f1_max = max([(max(a), a.index(max(a))) for a in svm_values])

##Pregunta j
#Maximos f1_scores obtenidos por metodo
f1_max_scores_df = pd.DataFrame(data = {"Naive Bayes": max(naive_values),
                                        "MultinomialNB": max(multinomial_values),
                                        "Logistic Regression": logit_f1_max[0],
                                        "SVM": svm_f1_max[0]},
                                index = np.arange(4))
sns.pointplot(data=f1_max_scores_df)
sns.plt.show()

#f1_scores para naive bayes
column_names = ["WE2,t","WE2,f","WE,t","WE,f"]
f1_naive_df = pd.DataFrame(data = np.array(naive_values).reshape(1,len(naive_values)), index = [0], columns = column_names)
sns.pointplot(data=f1_naive_df)
sns.plt.show()

train_naive_df = pd.DataFrame(data = np.array(naive_train_acc).reshape(1,len(naive_values)), index = [0], columns = column_names)
test_naive_df = pd.DataFrame(data = np.array(naive_test_acc).reshape(1,len(naive_values)), index = [0], columns = column_names)

sns.pointplot(data=train_naive_df)
sns.pointplot(data=test_naive_df)
sns.plt.show()
#f1_scores para multinomial

f1_multinomial_df = pd.DataFrame(data = np.array(multinomial_values).reshape(1,len(multinomial_values)), index = [0], columns = column_names)
sns.pointplot(data=f1_multinomial_df)
sns.plt.show()

train_multinomial_df = pd.DataFrame(data = np.array(naive_train_acc).reshape(1,len(naive_values)), index = [0], columns = column_names)
test_multinomial_df = pd.DataFrame(data = np.array(naive_test_acc).reshape(1,len(naive_values)), index = [0], columns = column_names)

sns.pointplot(data=train_multinomial_df)
sns.pointplot(data=test_multinomial_df)
sns.plt.show()
#f1_scores para logistic regression

f1_logit_df = pd.DataFrame(data = np.array(logit_values).T, columns = column_names)
f1_logit_df["cs"] = logit_cs
#Seaborn se comporta mejor con dataframes en "long form"
f1_logit_df_melted = pd.melt(f1_logit_df,id_vars=["cs"], value_vars = column_names)
sns.pointplot(x = "cs", y = "value", data = f1_logit_df_melted, hue = "variable")
sns.plt.show()

#f1_scores para linear svm

f1_svm_df = pd.DataFrame(data = np.array(svm_values).T, columns = column_names)
f1_svm_df["cs"] = svm_cs
#Seaborn se comporta mejor con dataframes en "long form"
f1_svm_df_melted = pd.melt(f1_svm_df,id_vars=["cs"], value_vars = column_names)
sns.pointplot(x = "cs", y = "value", data = f1_svm_df_melted, hue = "variable")
sns.plt.show()
