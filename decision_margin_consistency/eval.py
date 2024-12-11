import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kurtosis

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from pdb import set_trace

class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        """
        Wraps a dataset to include index and file paths (optional).
        
        Args:
            dataset (Dataset): The original dataset.
            filepaths (list or None): A list of file paths corresponding to dataset samples.
        """
        self.dataset = dataset
        self.filepaths = self.get_filepaths(dataset)
    
    def get_filepaths(self, dataset):
        imgs = (dataset.imgs if hasattr(dataset, "imgs") 
                else dataset.samples if hasattr(dataset, "samples") 
                else dataset._samples if hasattr(dataset, "_samples")
                else [])
        filepaths = [(os.path.sep).join(f.split(os.path.sep)[-3:]) for f,_ in imgs]

        return filepaths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if isinstance(sample, dict):
            image = sample['image']
            label = sample['label']
            path = sample['path']
        else:
            image = sample[0]
            label = sample[1]
            path = self.filepaths[index] if self.filepaths else None
            
        return image, label, index, path
    
@torch.no_grad()    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    act, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    corrects = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        corrects.append(correct[:k].any(dim=0).float())
    return pred, act, corrects, res
    
@torch.no_grad()
def validation(model, dataset, topk=(1,5), batch_size=250,
               num_workers=len(os.sched_getaffinity(0)) - 1, 
               shuffle=False, pin_memory=True, meta={}):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction='none')
    dataset = DatasetWrapper(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=shuffle, pin_memory=pin_memory)
    
    model.eval()
    results = defaultdict(list)
    count = 0
    for i, batch in enumerate(tqdm(dataloader)):
        images = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        index = batch[2].tolist()
        filenames = batch[3]
        batch_size = images.shape[0]
        output = model(images)
                
        target = target.to(output.device)
        loss = criterion(output, target)

        preds, act, accuracies, _ = accuracy(output, target, topk=topk)     

        # maximum target and non-target activation
        target_activations = torch.gather(output, 1, target.view(-1,1)).squeeze()
        mask = torch.ones_like(output, dtype=bool).scatter_(1, target.view(-1, 1), False)
        non_target_output = output[mask].view(output.shape[0], -1)
        max_nontarget_activation = non_target_output.max(dim=1).values
        decision_margin = (target_activations-max_nontarget_activation) / math.sqrt(2)
        
        results['index'] += index            
        results['filename'] += filenames
        results['target_label'] += target.tolist()        
        results['predicted_label'] += preds[0].tolist()
        results['top5_labels'] += preds.t().tolist()
        results['top5_outputs'] += act.tolist()
        results['target_act'] += target_activations.tolist()
        results['max_nontarget_act'] += max_nontarget_activation.tolist()
        results['decision_margin'] += decision_margin.tolist()
        results['loss'] += loss.tolist()
        
        for idx,k in enumerate(topk):            
            if k==1:
                results['is_correct'] += accuracies[idx].tolist()
            else:
                results[f'top{k}'] += accuracies[idx].tolist()
        
    # combine rawdata (image-by-image results) and summary into output dict
    df = pd.DataFrame(results)
    for idx,(key,val) in enumerate(meta.items()):
        df.insert(idx, key, val)
        
    # leaderboard summary    
    summary = dict(**meta, **dict(        
        dm_mean=df.decision_margin.mean(),
        dm_min=df.decision_margin.min(),
        dm_max=df.decision_margin.max(),
        dm_kurtosis=kurtosis(df.decision_margin),
    ))
    for idx,k in enumerate(topk):
        if k==1:
            summary[f'top{k}'] = df['is_correct'].mean()*100
        else:
            summary[f'top{k}'] = df[f'top{k}'].mean()*100
    
    summary = pd.DataFrame(summary, index=[0])

    return df, summary