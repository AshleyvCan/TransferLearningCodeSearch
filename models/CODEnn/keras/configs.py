
def config_JointEmbeddingModel():   
    config = {
        'data_params':{
      
            

            'train_methname':'python.train.name.pkl',

            'train_tokens':'python.train.code.pkl',
            'train_desc':'python.train.qt.pkl',
            
            #test data
            'valid_methname':'python.test.name.pkl',
            'valid_tokens':'python.test.code.pkl',
            'valid_desc':'python.test.qt.pkl',
            
            #use data (computing code vectors)
            'use_codebase': 'codesearchnet_1test_raw.json', 
            'use_methname': 'python.test.name.pkl', 

            'use_tokens': 'python.test.code.pkl',    
            
            'use_codevecs': 'vocab.desc.h5',  
                   
            #parameters
            'name_len': 6,

            'tokens_len':50,
            'desc_len': 30,

            'n_words_qt': 10002, 
           'n_words_name': 10002, 
            'n_words_code': 10002, 

            #parameters
            'methname_len': 6,
            'apiseq_len':30,
            'tokens_len':50,
            'desc_len': 30,
            'n_words': 10000,

            #vocabulary info
            'vocab_methname':  'python.name.vocab.json', #
            'vocab_tokens': 'python.code.vocab.json', 
            'vocab_desc': 'python.qt.vocab.json', 

            'size_testset': 22176, #for CodeSearchNet #27112 for StaQC
            'run_test_in_batch': 1,
        },               
        'training_params': {           
            'batch_size': 128,
            'chunk_size':-1,
            'nb_epoch': 101,
            'validation_split': 0.2,
            'optimizer': 'adam',
            'valid_every': 5,
            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'save_every': 5,
            'reload': 45 #epoch that the model is reloaded from . If reload=0, then train from scratch
        },

        'model_params': {
            'n_embed_dims': 100,
            'n_hidden': 400,
            'n_lstm_dims': 200, # * 2
            'init_embed_weights_methname': None,#'word2vec_100_methname.h5', 
            'init_embed_weights_tokens': None,#'word2vec_100_tokens.h5', 
            'init_embed_weights_desc': None,#'word2vec_100_desc.h5',           
            'margin': 0.05,
            'sim_measure':'cos',#similarity measure: gesd, cos, aesd
        }        
    }
    return config




