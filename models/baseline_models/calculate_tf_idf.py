import pickle
import json
import collections
import numpy as np
import math
from numpy.linalg import norm
import tqdm
import argparse
import os


# Reused from the CQIL repository (https://github.com/flyboss/CQIL)
def SuccessRate_Avg(real, predict, k):
    predict = predict[:k]
    sum = 0.0
    for val in real:
        try: index = predict.index(val)
        except ValueError: index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(real))

# Reused from the CQIL repository (https://github.com/flyboss/CQIL)
def MRR(real, predict, k):
    predict = predict[:k]
    sum = 0.0
    for val in real:
        try: index = predict.index(val)
        except ValueError: index = -1
        if index != -1: sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def retrieve_vocab(code_data):
    tokens_collection = []
    for i in range(len(code_data)):
        tokens_collection.extend(code_data[i]['body'].split(' '))
    code_vocab = dict(collections.Counter(tokens_collection).most_common(10000))

    tfidf = {}

    for i in range(len(code_data)):
        tokens = code_data[i]['body'].split(' ')
        token_counter = collections.Counter(tokens)

        for token in set(tokens):
            tf = token_counter[token]/len(code_vocab)
            if token in code_vocab.keys():
                df = code_vocab[token]
            else:
                df = 1
            idf = np.log(len(code_data)/(df+1))
            tfidf[i, token] = tf*idf
    code_vocab_tokens = list(code_vocab.keys())

    return code_vocab, code_vocab_tokens, tfidf

def extract_code_representations(tf_idf, code_vocab_tokens):
    dataset_matrix = np.zeros((len(code_data), len(code_vocab_tokens)))

    for i, j in tf_idf.keys():
        if j in code_vocab_tokens: 
            dataset_matrix[i][code_vocab_tokens.index(j)] = tf_idf[i,j] 
    
    return dataset_matrix

def extract_query_representations(query, code_vocab, code_vocab_tokens):
    tokens = query.split(' ')
    query_vector = np.zeros((len(code_vocab)))
    token_counter = collections.Counter(tokens)
    for token in set(tokens):
        tf = token_counter[token]/len(tokens)
        if token.lower() in code_vocab_tokens: 
            df = code_vocab[token.lower()]
        else:
            df = 1
        idf = math.log((len(code_data)+1)/(df+1))
        if token.lower() in code_vocab_tokens: 
            query_vector[code_vocab_tokens.index(token)] = tf*idf

    return query_vector

def compute_similarity(code_representations, query_vector): 
    return [np.dot(query_vector, code_snippet)/(norm(query_vector)*norm(code_snippet)) 
                    for code_snippet in code_representations]


def retrieve_top_predictions(similarities, real):
    top10_results = [similarities.index(v) for v in sorted(similarities, reverse=True)[:10]]
    sc1 = SuccessRate_Avg([real], top10_results, 1)
    sc5 = SuccessRate_Avg([real], top10_results, 5)
    sc10 = SuccessRate_Avg([real], top10_results, 10)
    MRR_score = MRR([real], top10_results, 10)

    return sc1, sc5, sc10, MRR_score, top10_results

def extract_relevant_data(working_dir, setting):
    path = working_dir + '/../../preprocesing/'

    if setting == 'csn':
        file_name = 'codesearchnet_1test.json'
    elif setting == 'staqc':
        file_name = 'test_data_staq.json'
    with open(path+ file_name, 'r') as f:
        code_data = json.load(f)
    return code_data

if __name__ == '__main__':
    parser = argparse.ArgumnetParser()
    parser.add_argument('--exper_setting', type = str, default = 'csn')
    settings = parser.parse_args()
    working_dir = os.getcwd()

    code_data = extract_relevant_data(working_dir, settings.exper_setting)
    all_scores = {'sc1':[], 'sc5': [], 'sc10': [], 'MRR_score': []}
    code_vocab, code_vocab_tokens, tf_idf = retrieve_vocab(code_data)

    for i in tqdm.tqdm(range(0, 4000)):

        query = code_data[i]['query']
        code_representations = extract_code_representations(tf_idf,  code_vocab_tokens) 
        query_vector = extract_query_representations(query, code_vocab, code_vocab_tokens)
        similarities = compute_similarity(code_representations, query_vector)
        sc1, sc5, sc10, MRR_score, _ = retrieve_top_predictions(similarities, i)
        all_scores['sc1'].append(sc1)
        all_scores['sc5'].append(sc5)
        all_scores['sc10'].append(sc10)
        all_scores['MRR_score'].append(MRR_score)

    with open('scores_tfidf.json', 'w') as f:
        json.dump(all_scores, f)
    