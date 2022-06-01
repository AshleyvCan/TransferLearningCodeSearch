import pickle
import json
import collections
import random
import os
import argparse

# from models.CODEnn.keras.main import parse_args

def import_json(file_name, path):
    with open(path + file_name, 'r') as f:
        data = json.load(f)
    return data

def convert_data_for_dcs(dataset, file_name, vocab, feature_name):
    pos = 0
    token_values = []
    indices = []
    non_exist = 0
    n_vocab = len(vocab.keys())
    for x in dataset:
        name = x[feature_name].split(' ')
        
        n_tokens = len(name)
        for token in name:
            if token in vocab.keys():
                token_values.append(vocab[token])
            else:
                non_exist += 1
                token_values.append(0)
        indices.append((n_tokens, pos))
        pos += n_tokens
 
    data = {'/phrases': token_values, '/indices': indices}
    with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def write_json(path, data, file_name):
    
    with open(path + file_name, 'w') as f:
        json.dump(data, f)




def create_vocab(dataset, min_freq):     
    name_txt, qt_txt, code_txt = [], [], []

    for j in range(len(dataset)):
        name = dataset[j]['name'].split(' ')
        code_tokens = dataset[j]['body'].split(' ')
        qt_tokens = dataset[j]['query'].split(' ')
    

        name_txt.extend(name)
        code_txt.extend(code_tokens)
        qt_txt.extend(qt_tokens)

    n_vocab = 10000
    vocab_name= {k:v for k, v in dict(collections.Counter(name_txt).most_common(n_vocab)).items() if v> min_freq} #
    vocab_code = {k:v for k, v in dict(collections.Counter(code_txt).most_common(n_vocab)).items() if v> min_freq}
    vocab_qt = {k:v for k, v in dict(collections.Counter(qt_txt).most_common(n_vocab)).items() if v> min_freq}
    vocab_name_dict = {keyword:i+1 for i, keyword in enumerate(vocab_name)}
    vocab_name_dict['UNK'] = 0
    vocab_code_dict = {keyword:i+1 for i, keyword in enumerate(vocab_code)}
    vocab_code_dict['UNK'] = 0
    vocab_qt_dict = {keyword:i+1 for i, keyword in enumerate(vocab_qt)}
    vocab_qt_dict['UNK'] = 0

    return vocab_name_dict, vocab_code_dict, vocab_qt_dict

def import_relevant_data(working_dir, setting):
    
    if setting == 'csn':
        file_train = '/codesearchnet_1train.json'
        file_val = '/codesearchnet_1valid.json'
        file_test = '/codesearchnet_1test.json'
        
    elif setting == 'staqc':
        file_train = '/train_data_staq.json'
        file_val = '/val_data_staq.json'
        file_test = '/test_data_staq.json'
    
    elif setting == 'csn_t':
        file_train = '/codesearchnet_1train.json'
        file_val = '/codesearchnet_1valid.json'
        file_test = '/test_data_staq.json'

    elif setting == 'staqc_t':
        file_train = '/train_data_staq.json'
        file_val = '/val_data_staq.json'
        file_test = '/codesearchnet_1test.json'

    
    trainset = import_json(file_train, working_dir)
    valset = import_json(file_val, working_dir)
    testset = import_json(file_test, working_dir)
    
    return trainset, valset, testset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_setting", type=str, default='csn')
    settings = parser.parse_args()
    working_dir = os.getcwd()

    trainset, valset, testset = import_relevant_data(working_dir, settings.exper_setting)

    vocab_name, vocab_code, vocab_qt = create_vocab(trainset, 0)

    path_save =  working_dir+ "/dcs/"

    write_json(path_save, vocab_name, 'python.name.vocab.json')
    write_json(path_save, vocab_code, 'python.code.vocab.json')
    write_json(path_save, vocab_qt, 'python.qt.vocab.json')

    
    convert_data_for_dcs(trainset, path_save+'python.train.name', vocab_name, 'name')
    convert_data_for_dcs(valset, path_save+'python.val.name', vocab_name, 'name')
    convert_data_for_dcs(testset, path_save+'python.test.name', vocab_name, 'name')


    convert_data_for_dcs(trainset, path_save+'python.train.code', vocab_code, 'body')
    convert_data_for_dcs(valset, path_save+'python.val.code', vocab_code, 'body')
    convert_data_for_dcs(testset, path_save+'python.test.code', vocab_code, 'body')


    convert_data_for_dcs(trainset, path_save+'python.train.qt', vocab_qt, 'query')
    convert_data_for_dcs(valset, path_save+'python.val.qt', vocab_qt, 'query')
    convert_data_for_dcs(testset, path_save+'python.test.qt', vocab_qt, 'query')