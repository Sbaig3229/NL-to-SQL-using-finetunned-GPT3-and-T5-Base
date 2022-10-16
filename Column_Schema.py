import csv
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def column_name_pred(query,pathfile):
  with open(pathfile) as csv_file:
    csv_reader = csv.DictReader(csv_file)
    dict_from_csv = dict(list(csv_reader)[0])
    list_of_column_names = list(dict_from_csv.keys())
  tfidf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf.fit_transform(list_of_column_names)
  tfidf_matrix2 = tfidf.transform([query])
  results = cosine_similarity(tfidf_matrix,tfidf_matrix2)
  result=results.ravel()
  indices=np.where(result >= 0.5)
  for i in indices:
    if i.size > 0:
      indices= i[0]
      matched_column=list_of_column_names[indices]
    else:
      matched_column=[]
  word_pred=matched_column
  newWords = []
  for word in query.split():
    result = difflib.get_close_matches(word, [word_pred.lower()], n=1)
    newWords.append(word_pred if result else word)
  result = ' '.join(newWords)
  return result, word_pred

def data_matching(query,pathfile):
  query,column_name=column_name_pred(query,pathfile)
  if len(column_name)>0:
    data=pd.read_csv(pathfile)
    data_pred_column=data[column_name].astype('string')
    tfvect = TfidfVectorizer(stop_words='english')
    tfvect_matrix = tfvect.fit_transform(data_pred_column)
    tfvect_matrix2 = tfvect.transform([query])
    results = cosine_similarity(tfvect_matrix,tfvect_matrix2)
    result=results.ravel()
    indices=np.where(result >= 0.5)
    for i in indices:
      if i.size > 0:
        indices= i[0]
        matched_values=data_pred_column[indices]
      else:
        matched_values=[]
    data_pred=matched_values
    newWords = []
    for word in query.split():
      result = difflib.get_close_matches(word, [data_pred], n=1)
      newWords.append(result[0] if result else word)
    result = ' '.join(newWords)
    return result