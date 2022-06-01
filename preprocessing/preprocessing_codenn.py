from preprocessing_csn import import_jsonl, split_methodname, remove_punc, check_empty_pairs
from sklearn.model_selection import train_test_split
import pickle
import argparse
from preprocessing_staqc import split_data
import os

def write_data(data, split_name, path):
    with open(path + '/' +split_name + '.txt', 'w', encoding= 'utf-8') as f:
        f.write(data)

def remove_docstring(code, docstr):
    replace_with_empty = [docstr, '"""\n    ', '"""\n', '''''''', '"""\r    ', '"""\r']
    replace_with_space = ['\n', '\t', '\r']
    for r in replace_with_empty: code = code.replace(r, '')
    for r in replace_with_space: code = code.replace(r, ' ')
    
    return code
    
def extract_relevant_data(code_snippets):   
    code_info = [{'query': ' '.join(filter(None, [remove_punc(qt) for qt in code_snippet['docstring_tokens']])),
            'body': remove_docstring(code_snippet['code'], code_snippet['docstring'])} 
            for code_snippet in code_snippets ]
    return [code for code in code_info if check_empty_pairs(code, ['query','body']) == 2]


def extract_data_csn(split_name, path_load, number_of_files = 1):
    full_dataset = []
    for i in range(number_of_files):
    
        code_snippets = import_jsonl(path_load + split_name + "/python_" + split_name + "_" + str(i) +".jsonl.gz")
        sub_dataset = extract_relevant_data(code_snippets)
        full_dataset.extend(sub_dataset)

    return full_dataset
        

def extract_data_staqc(path, filename_code, filename_qt):
    with open(path +'/' + filename_code, 'rb') as f:
        code_data = pickle.load(f)

    with open(path +'/' + filename_qt, 'rb') as f:
        qt_data = pickle.load(f)
    
    code_functions = []
    for k, code_snippet in code_data.items():
        code_info = {}      
        if isinstance(k, int):
            indx = k
        else:
            indx = k[0]
        code_info['body'] = code_snippet   
        code_info['query'] = qt_data[indx]
        code_functions.append(code_info)
    return code_functions

def convert_data_to_string(data, indx_code, indx_qt, start_indx):
    data_in_str = []
    for i, code in enumerate(data): 
        data_in_str.append(str(indx_qt[i+start_indx])+ '\t'+ str(indx_code[i+start_indx]) + '\t'+ code['query'].replace('\r', '') + '\t'+ str(code['body'].replace('\n', ' ')) +'\t0')
    
    return data_in_str

def import_relevant_data(working_dir, setting):
    
    if setting == 'csn':
        path_load = working_dir + "/python/final/jsonl/"
       
        train_data = extract_data_csn("train", path_load, number_of_files = 14)
        val_data = extract_data_csn("valid", path_load)
        test_data = extract_data_csn("test", path_load)


    elif setting == 'staqc':
        filename_code_multiple = 'python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle'
        filename_qt_muttiple = 'python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle'

        filename_code_single = 'python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle'
        filename_qt_single = 'python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle'

        path = working_dir + '/StaQC/'
        code_functions = extract_data_staqc(path,filename_code_multiple, filename_qt_muttiple)
        code_functions.extend(extract_data_staqc(path, filename_code_single, filename_qt_single))
        train_data, val_data, test_data = split_data(code_functions)
    
    return train_data, val_data, test_data

def save_test_batches(test_set, path_save, number_of_batches = 4, k = 1000):

    for i in range(0, number_of_batches, k):
        sub_test_set = test_set.split('\t0\n')[i:i+k]

        dev = '\t0\n'.join(sub_test_set) + '\t0'
        ref = '\n'.join([line.split('\t')[0] + '\t' + line.split('\t')[2] for line in sub_test_set])

        write_data(dev, 'dev', path_save)
        write_data(ref, 'ref', path_save)

def convert_data_for_txt(train_data, val_data, test_data):
    total_data_len = len(train_data) + len(val_data) + len(test_data)
    indx_code = list(range(1, total_data_len + 1))
    indx_qt = list(range(max(indx_code) + 1, max(indx_code) + total_data_len + 1))

    train_set =  '\n'.join(convert_data_to_string(train_data, indx_code, indx_qt, 0))
    val_set =  '\n'.join(convert_data_to_string(val_data, indx_code, indx_qt, len(train_data)))
    test_set =  '\n'.join(convert_data_to_string(test_data, indx_code, indx_qt, len(train_data) + len(val_data)))
    return train_set, val_set, test_set


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_setting", type=str, default='csn')
    settings = parser.parse_args()
    working_dir = os.getcwd()

    train_data, val_data, test_data = import_relevant_data(working_dir, settings.exper_setting)

    train_set, val_set, test_set = convert_data_for_txt(train_data, val_data, test_data)

    path_save = working_dir

    write_data(train_set, 'train', path_save)
    write_data(val_set, 'valid', path_save)
    write_data(test_set, 'test', path_save)
    save_test_batches(test_set, path_save)
