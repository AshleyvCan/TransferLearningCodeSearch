import logging
import math
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from CQIL_dataset import TrainDataset, TestDataset, ValDataset
from utils.metrics import get_metrics_scores, metrics_scores_to_dict, metrics_scores_to_dict_singles

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from torch.optim.lr_scheduler import LambdaLR

from utils.util import save_model


class CQILHelper:
    def __init__(self, config):
        self.load_dataset(config)
        self.device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")

    def load_dataset(self, config):
        if config['type'] == 'recom':
            self.eval_dataset = TestDataset(config, dataset_type='eval')
            print(1)
        else:
            self.train_dataset = TrainDataset(config)
            self.valid_dataset = ValDataset(config, dataset_type='valid')

    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=.5,
                                        last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def train(self, config, model):

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        model.train()
        data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=config['batch_size'],
                                                  shuffle=True, drop_last=True, num_workers=0)

        emb_layers = nn.ModuleList([model.query_embedding, model.name_embedding, model.body_embedding])
        emb_layers_paras = list(map(id, emb_layers.parameters()))
        base_paras = filter(lambda p: id(p) not in emb_layers_paras, model.parameters())
        optimizer_grouped_parameters = [
            {'params': base_paras},
            {'params': emb_layers.parameters(), 'lr': config['emb_learning_rate']}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'],
                                      eps=1e-8)

        scheduler = self.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config['warmup_steps'],
            num_training_steps=len(data_loader) * config['nb_epoch'])

        n_iters = len(data_loader)
        itr_global = config['reload'] + 1

        best_mrr = 0
    
        for epoch in range(int(config['reload'] / n_iters) + 1, config['nb_epoch'] + 1):
            losses = []
            
            for batch in data_loader:
                model.train()

                batch_gpu = [tensor.to(self.device) if i not in [0,1,2,3,4] else tensor.to(self.device).long() for i, tensor in enumerate(batch)]
                
                loss = model(*batch_gpu)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                losses.append(loss.item())

                if itr_global % config['log_every'] == 0:
                    info = f'epo:[{epoch}/{config["nb_epoch"]}] itr:[{itr_global % n_iters}/{n_iters}] ' \
                           f'Loss={np.mean(losses)} learning rate: {optimizer.param_groups[0]["lr"]}'
                    logger.info(info)
 
                    losses = []

                itr_global = itr_global + 1

                if itr_global % config['valid_every'] == 0:
                    logger.info("\nvalidating..")
                    mrr = self.test(config, model, self.valid_dataset)
                    if mrr > best_mrr:
                        best_mrr = mrr
                        print(f'best mrr is {best_mrr}, save model...')
                        save_model(model, config['model_save_dir'], itr_global)

                if itr_global % config['save_every'] == 0:
                    save_model(model, config['model_save_dir'], itr_global)
            print('loop done')
        logger.info('test on eval dataset')
        self.test(config, model, self.valid_dataset)
        save_model(model, config['model_save_dir'], str(itr_global) + "_finnal")
    
    def test(self, config, model, dataset, batch_size=1000, top_results=10):
        model.eval()

        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                  shuffle=False, drop_last=True, num_workers=0)
        metrics_scores = []
        with torch.no_grad():
            for batch in data_loader:
                
                query_batch, name_batch, body_batch = [_.to(self.device) for _ in batch[0:3]]
                name_lex_batch, body_lex_batch = batch[3], batch[4]

                for i in tqdm(range(batch_size)):
                    query = query_batch[i].expand(batch_size, config['max_query_len'])
                    name_lex = name_lex_batch[i].to(self.device)
                    body_lex = body_lex_batch[i].to(self.device)
                    sims = model.compute(query.long(), name_batch.long(), body_batch.long(), name_lex,
                                         body_lex).cpu().detach().numpy().flatten()

                    negsims = np.negative(sims)
                    predict = np.argsort(negsims)
                    predict = predict[:top_results]
                    predict = [int(k) for k in predict]
                    real = [i]
                    metrics_scores.append(get_metrics_scores(real, predict))

            metrics_dict = metrics_scores_to_dict(metrics_scores)
            logger.info(str(metrics_dict))
            metrics_dict_all = metrics_scores_to_dict_singles(metrics_scores)
            with open('scores.json', 'w') as f:
                json.dump(metrics_dict_all, f)
        return metrics_dict['MRR']
    

    def retrieve_recommendation(self, config, model, query_data, dataset, batch_size=1000, top_results=10, data_eval = None):
        #batch_size=990
        model.eval()
        
        
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                  shuffle=False, drop_last=True, num_workers=0)
        

        metrics_scores = []
        
        with torch.no_grad():
            
            for batch in data_loader:
               
                name_batch, body_batch = [_.to(self.device) for _ in batch[0:2]]

                torch.set_printoptions(edgeitems=20)
                
                for i in tqdm(range(1)):
                    #print(torch.from_numpy(query_data[0][i]))
                    query = torch.from_numpy(query_data[0][i]).expand(batch_size, config['max_query_len']).to(self.device)
                    name_lex = torch.from_numpy(query_data[1][i]).to(self.device)
                    body_lex = torch.from_numpy(query_data[2][i]).to(self.device)
                    #print(name_lex.shape, body_lex.shape)
                    sims = model.compute(query.long(), name_batch.long(), body_batch.long(), name_lex,
                                         body_lex).cpu().detach().numpy().flatten()
                    #print(len(sims))
                    negsims = np.negative(sims)
                    predict = np.argsort(negsims)
                    predict = predict[:top_results]
                    predict = [int(k) for k in predict]
                    real = [i]

                    print(i)
                    print(predict)
                    for j in predict[:2]:
                        print(data_eval[j]['body'])
                    metrics_scores.append(get_metrics_scores(real, predict))
            print(len(query_data[0]))   
            metrics_dict = metrics_scores_to_dict(metrics_scores)
            logger.info(str(metrics_dict))

        return metrics_dict['MRR']
