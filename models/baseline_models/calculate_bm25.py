from rank_bm25 import BM25Okapi
import json
from metrics import get_metrics_scores, metrics_scores_to_dict_singles
import argparse
import os

def extract_corpus_and_queryes(data):
    search_corpus = [code_info['body'] for code_info in data]
    search_queries = [code_info['query'] for code_info in data]
    return search_corpus, search_queries

def calculate_bm25_corpus(search_corpus):
    preprocessed_code = [code_snippet.split(" ") for code_snippet in search_corpus]

    return BM25Okapi(preprocessed_code)

def extract_relevant_data(workdir, setting):
    path = workdir + '/../preprocessing/'
    if setting == 'staqc':
        file_name =  'test_data_staqc.json'
    elif setting == 'csn':

        file_name =  'codesearchnet_1test.json'
       
    with open(path + file_name, 'r') as f:
        code_data = json.load(f)
    return code_data

def compute_similarities(bm25, search_queries, search_corpus):
    scores = []
    for i in range(len(search_queries)):
        preprocessed_query = search_queries[i].split(" ")
        recom = bm25.get_top_n(preprocessed_query, search_corpus, n=10)
        predict = [search_corpus.index(code) for code in recom]
        scores.append(get_metrics_scores([i], predict))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exper_setting', type=str, default = 'csn')
    settings = parser.parse_args()
    working_dir = os.getcwd()
    code_data = extract_relevant_data(working_dir, settings.exper_setting)

    search_corpus, search_queries =extract_corpus_and_queryes(code_data)
    bm25 = calculate_bm25_corpus(search_corpus)
    scores=compute_similarities(bm25, search_queries, search_corpus)

    metrics_dict = metrics_scores_to_dict_singles(scores)
    with open('scores_bm25_csn.json', 'w') as f:
        json.dump(metrics_dict, f)
