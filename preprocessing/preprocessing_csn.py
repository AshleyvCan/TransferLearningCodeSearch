import json
import gzip
# from posixpath import split
import re
import os

def import_jsonl(file_name):
    """
    Imports jsonl files into a list
    """
    file_content = []
    with gzip.open(file_name, 'r') as file: 
        for line in file:
            file_content.append(json.loads(line))
    return file_content 


def write_data(data, split_name, path):
    """
   Writes data to json file
    """
    with open(path + '/codesearchnet_1'+ split_name + '.json', 'w', encoding= 'utf-8') as outfile:
        json.dump(data, outfile)


def split_methodname(name):
    """
    Extract tokens out of method name with camel case and snake case.
    """
    tokens = [[name[0]]]
    
    name_list = list(name[1:])
    n_char = len(name_list)
    for i in range(n_char):
        if tokens[-1][-1].islower() and name_list[i].isupper():
            tokens.append([name_list[i]])
        elif name_list[i] in ['_', '.'] and i < (n_char-1):
            name_list[i+1] = name_list[i+1].upper()
        elif name_list[i-1] in ['_', '.'] and i == (n_char-1):
            tokens.append([name_list[i]])
        else:
            tokens[-1].append(name_list[i])
  
    return [''.join(k) for k in tokens]


def check_empty_pairs(code_pair, var_names):
    return sum([int(code_pair[k] != '') for k in var_names])


def remove_comments(code_snippets):
    """
    Removes docstring from code snippets
    """
    for i in range(len(code_snippets)):
        if code_snippets[i]['docstring'].split('\n\n')[0][-1] in ['.','!','?']:
           code_snippets[i]['docstring'] = code_snippets[i]['docstring'].split('\n\n')[0][:-1]
        else: code_snippets[i]['docstring'] = code_snippets[i]['docstring'].split('\n\n')[0]
    return code_snippets


def check_first_(word):
    """
    Removes '_' at start and end of string
    """
    while word != '' and word[0] == '_' : 
        word = word[1:]
    while word != '' and word[-1] == '_': 
        word = word[:-1]
    return word


def remove_punc(word):
    """
    Removes punctations from string
    """
    if word == '': return ''
    word = check_first_(word)
    
    if any(ch.isupper() for ch in word) or '_' in word: 
        word = ' '.join(split_methodname(word))
    return ' '.join(re.sub(r'[^\w\s]','', word.strip().lower()).split())


def extract_relevant_data(code_snippets):   
    """
    Extracts for each code snippet the name, query and body into a list of dicts
    Subsequently it also returns an additional list with dicst with solely the queries
    """
    return [{'name':  ' '.join(filter(None, [remove_punc(n) for n in split_methodname(code_snippet['func_name'])])),
            'query': ' '.join(filter(None, [remove_punc(qt) for qt in code_snippet['docstring_tokens']])),
            'body': ' '.join(filter(None, [remove_punc(c) for c in code_snippet['code_tokens']]))}
        for code_snippet in code_snippets if check_empty_pairs(code_snippet, ['docstring_tokens','code_tokens']) == 2], \
            [{'query': ' '.join(filter(None, [remove_punc(qt) for qt in code_snippet['docstring_tokens']]))} 
            for code_snippet in code_snippets if check_empty_pairs(code_snippet, ['docstring_tokens','code_tokens']) == 2]
        

def extract_data(split_name, path_load, path_save, number_of_files = 1):
    """
    Exacts all data from the CodeSearchNet files (https://github.com/github/CodeSearchNet) 
    and uses the function 'extract_relevant_data' to preprocess each file.
    """
    full_dataset = []
    for i in range(number_of_files):
        code_snippets = import_jsonl(path_load + split_name + "/python_" + split_name + "_" + str(i) +".jsonl.gz")
        sub_dataset, queries_data = extract_relevant_data(code_snippets)
        full_dataset.extend(sub_dataset)
        
    write_data(full_dataset, split_name, path_save)
    write_data(queries_data, split_name +'query', path_save)


def extract_rawcode(split_name, path_load, path_save, number_of_files = 1):
    """
    Extract solely raw code, which can be used if the datasets is desired to be used subsequently as search corpus.
    These raw code snippets should be recommended to the uses instead of the preprocessed code snippets.
    """
    full_dataset = []
    for i in range(number_of_files):
        
        code_snippets = import_jsonl(path_load + split_name + "/python_" + split_name + "_" + str(i) +".jsonl.gz")
        code_snippets = [code_snippet['code'] for code_snippet in code_snippets]

        full_dataset.extend(code_snippets)
    write_data(full_dataset, split_name + '_raw', path_save)
        

if __name__ == "__main__":
    working_dir = os.getcwd()

    path_load = working_dir + "/python/final/jsonl/"
    # path_save = 'C:/Users/Gebruiker/Documents/Graduation_Internship/CodeSearchNet'
    extract_rawcode("test", path_load, working_dir)
    extract_rawcode("valid", path_load, working_dir)
    extract_data("train", path_load, working_dir, number_of_files = 14)
    extract_data("valid", path_load, working_dir)
    extract_data("test", path_load, working_dir)