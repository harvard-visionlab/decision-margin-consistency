import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations, product
from natsort import natsorted
from collections import defaultdict

def decision_margin_consistency(values1, values2, sim_fun=pearsonr):
    '''
        decision_margin_consistency is simply the correlation between
        two sets of decision margin scores. 
        
        values1 and values2 are assumed to be sets of decision-margin-scores
        that have already been aligned/paired.
    '''
    sim = sim_fun(values1, values2)
    if isinstance(sim, (tuple, list)):
        sim = sim[0]
    return sim

def error_consistency(expected_consistency, observed_consistency):
        """Return error consistency as measured by Cohen's kappa."""

        assert expected_consistency >= 0.0
        assert expected_consistency <= 1.0
        assert observed_consistency >= 0.0
        assert observed_consistency <= 1.0

        if observed_consistency == 1.0:
            return 1.0
        else:
            return (observed_consistency - expected_consistency) / (1.0 - expected_consistency)
    
def expected_consistency(df1, df2):
    p1 = df1.is_correct.mean()
    p2 = df2.is_correct.mean()
    expected_consistency = p1 * p2 + (1 - p1) * (1 - p2)
    
    return expected_consistency, p1, p2

def observed_consistency(df1, df2):
    return (df1.is_correct == df2.is_correct).sum() / len(df1)

def _compute_error_consistency(df1, df2):
    expected_con, p1, p2 = expected_consistency(df1, df2)
    observed_con = observed_consistency(df1, df2)
    error_con = error_consistency(expected_con, observed_con)
    return expected_con, observed_con, error_con

def compute_consistency(df1, df2=None, subject_col='subj', condition_col='condition', item_col='filename'):
    '''compute error consistency and decision-margin consisteny between all pairs of subjects 
       scores are computed separately for each condition (i.e., value of `condition_col`)
        
        If only a single dataframe is passed
            - compare all subjects in df1 with each other
            - compute mean +/- 95% CI over all pairs
            
        If two dataframes are passed
            - compare each individual subject in df1 with each individual subject in df2.
            - compute mean +/- 95% CI for each subj in df1 (e.g., models) averaged over each subj in df2 (e.g., humans)        
    '''
    
    subs1 = df1[subject_col].unique()    
    if df2 is not None:
        # get all pairwise comparisions between subjects in df1 and df2
        subs2 = df2[subject_col].unique()
        pairs = list(product(subs1, subs2))
    else:   
        # get unique combinations of subjects in dfA
        subs2 = None
        pairs = list(combinations(subs1, 2))
        df2 = df1.copy() # subjects all come from some dataframe, so make copy of df1 to use for 2nd subject
        
    conditions1 = natsorted(df1[condition_col].unique())
    conditions2 = natsorted(df2[condition_col].unique())
    assert conditions1==conditions2, f"df1 and df2 must have the same conditions, got conditions1={conditions1}, conditiosn2={conditions2}"
    conditions = conditions1
    
    results = defaultdict(list)
    for pair_num, (sub1, sub2) in enumerate(pairs):
        for condition in conditions:
            # get subsets
            subset1 = df1[(df1[subject_col] == sub1) & (df1[condition_col]==condition)].sort_values(by=item_col).reset_index(drop=True)
            subset2 = df2[(df2[subject_col] == sub2) & (df2[condition_col]==condition)].sort_values(by=item_col).reset_index(drop=True)
            
            # make sure we have aligned subsets
            assert (subset1[item_col]==subset2[item_col]).all(), f"Oops, {item_col} not aligned for subset1, subset2"
            
            # compute error consistency
            expected_con, p1, p2 = expected_consistency(subset1, subset2)
            observed_con = observed_consistency(subset1, subset2)
            error_con = error_consistency(expected_con, observed_con)
            dmc = decision_margin_consistency(subset1.decision_margin,
                                              subset2.decision_margin)
                
            # update results
            results[condition_col].append(condition)
            results['pair_num'].append(pair_num)
            results['sub1'].append(sub1)
            results['sub2'].append(sub2)

            results['sub1_pct_correct'].append(p1)
            results['sub2_pct_correct'].append(p2)

            results['expected_consistency'].append(expected_con)
            results['observed_consistency'].append(observed_con)
            results['error_consistency'].append(error_con)
            results['decision_margin_consistency'].append(dmc)
    
    return pd.DataFrame(dict(results))