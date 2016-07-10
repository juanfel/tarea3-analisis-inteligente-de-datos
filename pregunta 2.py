import urllib
import pandas as pd
#pregunta a)
train_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.train"
test_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.dev"
train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
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
