import os
import json
from addict import Dict as DotDict
from numpy import random 
import numpy as np 
import pandas as pd
from collections import defaultdict 
from fastprogress import master_bar, progress_bar 
from pprint import pprint
from natsort import natsorted
import matplotlib.pyplot as plt
from glob import glob

from decision_margin_consistency.helpers.remote_data import get_remote_data_file

def missing(self, key):
    raise KeyError(key)
DotDict.__missing__ = missing

raw_data_urls = {
    "snr-edges-v1": 'https://s3.us-east-1.wasabisys.com/visionlab-members/alvarez/Projects/decision-margin-consistency/behavioral_experiments/snr-edges-v1/raw-data-b0166651b7.tar.gz'
}

def load_data(exp_name="snr-edges-v1", nTrials=160):
    cached_filename, data_dir = get_remote_data_file(raw_data_urls[exp_name])
    files = glob(os.path.join(data_dir, '*.txt'))
    print(f"Number of files: {len(files)}")
    df = None 
    for file in files:      
        with open(file) as f:
            data = json.loads(f.read())

        print("==> ", file)
        for key,value in data.items():
            if key != "trialData":
                if key == "totalTime":
                    print(f"{key}: {value/1000/60:4.2f}min")
                elif key in ['studyID','workerID','comments']:
                    print(f"{key}: {value}")

        df_ = pd.DataFrame(data['trialData'])
        assert len(df_)==nTrials, f"Ooops, expected {nTrials} trials, got {len(df_)}"
        df_['repeatNum'] = df_['trialsSinceLastPresented'].apply(lambda x: 1 if x==-1 else 2)
        df = pd.concat([df, df_])
    return df

def compute_summary(df):
    conds = sorted(df.condName.unique())
    categories = natsorted(df.targetCategory.unique())
    results = defaultdict(list)
    for condName in conds:  
        df_ = df[df.condName==condName]
        all_items = get_items(df_, categories)
        subjects = df_.workerID.unique()
        mb = master_bar(subjects)
        for subject in mb:
            for item in progress_bar(all_items, parent=mb):
                subset1 = df[(df.workerID==subject) & (df.targetFilename==item) * (df.repeatNum==1)]
                subset2 = df[(df.workerID==subject) & (df.targetFilename==item) * (df.repeatNum==2)]
                assert len(subset1)==1 
                assert len(subset2)==1
                results['condName'].append(condName)
                results['subject'].append(subject)
                results['item'].append(item)
                results['category'].append(subset1.iloc[0].targetCategory)
                results['correct1'].append(subset1.iloc[0].responseCorrect)
                results['correct2'].append(subset2.iloc[0].responseCorrect)
                results['correctAvg'].append( (subset1.iloc[0].responseCorrect+subset2.iloc[0].responseCorrect) / 2)
    results = pd.DataFrame(results)
    return results

def get_items(df, categories):
    all_items = []
    for category in categories:
        subset = df[df.targetCategory==category]
        items = natsorted(subset.targetFilename.unique())
        all_items += items
    return all_items

def compute_error_consistency(acc1, acc2):
    n = len(acc1)

    # proportion of same responses  
    c_obs = (acc1 == acc2).sum() / n

    # expected overlap by chance
    p1 = acc1.mean() 
    p2 = acc2.mean()
    c_exp = p1 * p2 + (1 - p1) * (1 - p2 )

    # boostrap confidence intervals                                                    
    ci = bootstrap_ci(acc1, acc2, n_experiments=10000)

    # bounds 
    lower, upper = compute_cobs_bounds(c_exp)

    # cohen's kappa
    k = compute_k(c_obs, c_exp)

    return DotDict({
        "c_obs": c_obs,
        "c_exp": c_exp,
        "bootstrap": ci,
        "bounds": {
            "lower": lower,
            "upper": upper
        },
        "k": k 
    })  

def compute_k(c_obs, c_exp):
    k = (c_obs - c_exp) / (1 - c_exp)
    return k
  
def compute_ci(values, alpha=.95):  
    ordered = np.sort(values)
    lower_bound = ((1-alpha)/2) * 100
    upper_bound = (alpha+((1-alpha)/2)) * 100
    lower = max(0., np.percentile(ordered, lower_bound))
    upper = min(1., np.percentile(ordered, upper_bound))
    return lower, upper

def compute_cobs_bounds(cexp):
    '''compute the bounds of consistency_observed given a specific value of consistency_expected'''
    if cexp <= .5:
        lower_bound = 0
        upper_bound = 1 - np.sqrt(1-2*cexp)
    else:
        lower_bound = np.sqrt(2*cexp - 1)
        upper_bound = 1

    return lower_bound, upper_bound

def compute_k_bounds(cexp):
    '''compute the bounds of k given a specific value of consistency_expected'''
    if cexp <= .5:
        lower_bound = -cexp / (1-cexp)
        upper_bound = (1 - np.sqrt(1 - 2*cexp) - cexp) / (1 - cexp)
    else:
        lower_bound = (np.sqrt(2*cexp - 1) - cexp) / (1 - cexp)
        upper_bound = 1

    return lower_bound, upper_bound 

def gen_samples(n_trials, p, n_experiments=10000):
    '''returns simulated n_trials x n_samples'''
    return random.binomial(n=1, p=p, size=(n_trials, n_experiments))

def bootstrap_ci(acc1, acc2, n_experiments=10000, alpha=.95):
    samples1 = gen_samples(len(acc1), acc1.mean(), n_experiments)
    samples2 = gen_samples(len(acc2), acc2.mean(), n_experiments)

    # proportion of same responses  
    c_obs = (samples1 == samples2).mean(axis=0)

    # expected overlap by chance
    p1 = acc1.mean() 
    p2 = acc2.mean()
    c_exp = p1 * p1 + (1 - p1) * (1 - p2 )

    # cohen's kappa
    k = (c_obs - c_exp) / (1 - c_exp)

    c_obs_lower, c_obs_upper = compute_ci(c_obs, alpha=alpha)
    k_lower, k_upper = compute_ci(k, alpha=alpha)

    ci = DotDict({
        "n_trials": len(acc1),
        "n_experiments": n_experiments,
        "alpha": alpha,
        "actual_acc": {
            "acc1": acc1.mean(),
            "acc2": acc2.mean(),
        },
        "simulated_acc": {
            "acc1": samples1.mean(),
            "acc2": samples2.mean(),
        },
        "c_obs": {
            "avg": c_obs.mean(),
            "lower": c_obs_lower,
            "upper": c_obs_upper,
        },
        "k": {
            "avg": k.mean(),
            "lower": k_lower,
            "upper": k_upper
        }
    })

    return ci