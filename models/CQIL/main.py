import argparse
import logging

import torch

from CQIL import CQIL
from CQIL_helper import CQILHelper
from config import get_config
from utils.util import load_model
import json
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from CQIL_dataset import TestDataset, ValDataset, get_queries
from pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser('Train and Valid and Eval CQIL')
    parser.add_argument('--mode', choices=['train', 'valid', 'eval', 'recom'], default='train',
                        help='The mode to run. '
                             ' the `train` mode trains CQIL;'
                             ' the `valid` mode tests CQIL in the valid set;'
                             ' the `eval` mode tests CQIL in the eval set.'
                             ' the `recom` mode tests CQIL in the eval set.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = get_config()
    logger.info(config)
    config['type'] = args.mode
    CQIL_helper = CQILHelper(config)

    logger.info('Constructing Model...')
    model = CQIL(config)
   
    logger.info(model)
    if config['reload'] > 0:
        logger.info('load model')
        load_model(model, config['model_filepath'])
    
    model = model.to(torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('total parameters: ' + str(total_params))

    
    if args.mode == 'train':
        CQIL_helper.train(config, model)
    elif args.mode == 'valid':
        valid_dataset = ValDataset(config, dataset_type='valid')
        CQIL_helper.test(config, model, valid_dataset)

    elif args.mode == 'eval':
        data_eval = json.loads(open(config['data_dir'] + config['eval'], 'r').readline())
        eval_dataset = ValDataset(config, dataset_type='eval')
        #query_data = get_queries(config)
    
        CQIL_helper.test(config, model, eval_dataset)
    
    elif args.mode == 'recom':
        pipeline = Pipeline(config)
        data_eval = json.loads(open(config['data_dir'] + config['eval'], 'r').readline())
        paths = json.loads(open(config['data_dir'] + config['paths'], 'r').readline())    
        raw_eval = json.loads(open(config['data_dir'] + config['eval2'], 'r').readline())
        while True:
            eval_dataset = TestDataset(config, dataset_type='recom', dataset_file = data_eval)
            pipeline.run(config)
            query_data = get_queries(config)
            number_of_results = input("How many recommendations would you like to receive? (max 10): ")
            CQIL_helper.retrieve_recommendation(config, model, query_data, eval_dataset, 10000 ,int(number_of_results), paths= paths,  raw_eval = raw_eval)

            stop = input("Do you want to quit this session? (y/n): ")

            if stop.lower() == 'y' or stop.lower() == 'yes': break
