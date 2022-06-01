import json
import pickle
import re
from preprocessing_csn import split_methodname, remove_punc
#import sklearn
from sklearn.model_selection import train_test_split
import os


def convert_stacq_data(code_data, qt_data):
    code_functions = []
    for k, code_snippet in code_data.items():
        code_info = {}      
        if isinstance(k, int):
            indx = k
        else:
            indx = k[0]

        code_info['name'] = ''
        if code_snippet.count('def ') > 0:
            split_on_def = code_snippet.split('def ')
            
            for i in range(1, code_snippet.count('def ')+1):
                #if split_on_def[i] in ['', ' ', '\n']:
                if split_on_def[i].split('(')[0] != '':
                    code_info['name'] += ' '.join([remove_punc(kword) for kword in split_methodname(split_on_def[i].split('(')[0])]) + ' '

        if code_snippet.count('class ') > 0: 
            split_on_class = code_snippet.split('class ')    
            for i in range(1, code_snippet.count('class ')+1):
                if split_on_class[i].split('(')[0] != '':
                    code_info['name'] += ' '.join([remove_punc(kword) for kword in split_methodname(split_on_class[i].split('(')[0])]) + ' '
        
        code_info['name'] = re.sub(' +', ' ', code_info['name'].strip()) #''.join(code_info['name'].split()) #code_info['name'].strip()
            
        code_info['query'] = remove_punc(qt_data[indx])
        code_info['body'] = ' '.join(filter(None, [remove_punc(frag) for frag in re.findall(r"[\w']+|[.,!?;]", code_snippet)]))           
        code_functions.append(code_info)
    return code_functions

def split_data(code_functions):
    unique_code_functions = [i for n, i in enumerate(code_functions) if i not in code_functions[:n]] 

    train_data, test_data = train_test_split(unique_code_functions, test_size = 0.2, random_state = 42)
    val_data, test_data = train_test_split(test_data, test_size = 0.5, random_state = 42)

    return train_data, val_data, test_data

if __name__ == "__main__":
    working_dir = os.getcwd()
    path = working_dir + '/StaQC'

    with open(path+'/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle', 'rb') as f:
        code_stacq_multiple = pickle.load(f)

    with open(path+'/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle', 'rb') as f:
        qt_multiple = pickle.load(f)

    with open(path+'/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle', 'rb') as f:
        code_stacq_single = pickle.load(f)

    with open(path+'/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle', 'rb') as f:
        qt_single = pickle.load(f)



    code_functions = convert_stacq_data(code_stacq_multiple, qt_multiple)

    code_functions.extend(convert_stacq_data(code_stacq_single, qt_single))

    train_set, val_set, test_set = split_data(code_functions)
    # unique_code_functions = [i for n, i in enumerate(code_functions) if i not in code_functions[:n]] 

    # train_set, test_set = train_test_split(unique_code_functions, test_size = 0.2, random_state = 42)

    # val_set, test_set = train_test_split(test_set, test_size = 0.5, random_state = 42)


    with open('test_data_staq.json', 'w') as outfile:
        json.dump(test_set, outfile)

    with open('train_data_staq.json', 'w') as outfile:
        json.dump(train_set, outfile)

    with open('val_data_staq.json', 'w') as outfile:
        json.dump(val_set, outfile)


