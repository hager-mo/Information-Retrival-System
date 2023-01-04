import math
import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted

# first_part

# 3. Apply Stop words (except [in,to])
stopwords = stopwords.words('english')
stopwords.remove('in')
stopwords.remove('to')
stopwords.remove('where')

files_names = natsorted(os.listdir('files'))

document_of_terms = []
# 1.	Read 10 files (.txt)
for file in files_names:
    with open(f'files/{file}', 'r') as f:
        documents = f.read()
    # 2.	Apply tokenization
    terms = []
    splitted_documents = word_tokenize(documents)
    for word in splitted_documents:
        if word not in stopwords:
            terms.append(word)
    document_of_terms.append(terms)

print(document_of_terms)

# Second_part

# positional index
document_number = 1
positional_index = {}

for document in document_of_terms:
    for positional, term in enumerate(document):
        if term in positional_index:
            # increment frequency
            positional_index[term][0] = positional_index[term][0] + 1

            # Check if first time appears in the document or not
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)

            else:
                positional_index[term][1][document_number] = [positional]

        else:
            positional_index[term] = []
            positional_index[term].append(1)
            positional_index[term].append({})
            positional_index[term][1][document_number] = [positional]

    document_number += 1

print(positional_index)

# query

query = input("Enter query: ")
query = query.split()
matched_query_list = [[] for i in range(10)]

for word in query:
    for key in positional_index[word][1].keys():

        if matched_query_list[key-1] != []:
            if matched_query_list[key - 1][-1] == positional_index[word][1][key][0] - 1:
                matched_query_list[key - 1].append(positional_index[word][1][key][0])

        else:
            matched_query_list[key - 1].append(positional_index[word][1][key][0])

for pos, list in enumerate(matched_query_list, start=1):
    if len(list) == len(query):
        print(pos)

# third_part

#1. compute term frequency
all_words = []
for document in document_of_terms:
    for word in document:
        all_words.append(word)
# print(dict.fromkeys(all_words, 0))

def get_term_frequency(document):
    words_found = dict.fromkeys(all_words, 0)
    for word in document:
        words_found[word] += 1
    return words_found

term_freq = pd.DataFrame(get_term_frequency(document_of_terms[0]).values(), index=get_term_frequency(document_of_terms[0]).keys())

for i in range(1, len(document_of_terms)):
    term_freq[i] = get_term_frequency(document_of_terms[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print(term_freq)


def get_weighted_term_freq(n):
    if n > 0:
        return math.log(n) + 1
    return 0

for i in range(1, 11):
    term_freq['doc'+str(i)] = term_freq['doc'+str(i)].apply(get_weighted_term_freq)

print(term_freq)

#2. Compute IDF for each term

get_idf = pd.DataFrame(columns=['df', 'idf'])

for i in range(len(term_freq)):
    get_idf.loc[i, 'df'] = term_freq.iloc[i].values.sum()
    get_idf.loc[i, 'idf'] = math.log10(10.0 / float(term_freq.iloc[i].values.sum()))

get_idf.index = term_freq.index
print(get_idf)

#3.	Displays TF.IDF matrix

tf_idf = term_freq.multiply(get_idf['idf'], axis=0)
print(tf_idf)

#Doc length

document_length = pd.DataFrame()

def get_doc_len(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())

for column in tf_idf.columns:
    document_length.loc[0, column+' length'] = get_doc_len(column)

print(document_length)

#Normalize tf_idf

normalized_tf_idf = pd.DataFrame()

def get_normalized_tf_idf(col, n):
    if n > 0:
        return n / document_length[col + ' length'].values[0]
    return 0

for column in tf_idf.columns:
    normalized_tf_idf[column] = tf_idf[column].apply(lambda n: get_normalized_tf_idf(column, n))

print(normalized_tf_idf)

