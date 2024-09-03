# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import sys
import os
from IPython.display import display
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks

sys.path.append('../Algorithms')
from multicriteria import GaussianAHP, MCPM, SAC

# Set - path
path = '../Experiments/'
# Set - criteria
criteria = ['f1_macro', 'ratio_f1_macro', 'ratio_selection_rate', 'ratio_recall', 'ratio_odds']
# Set - datasets
datasets = ['adult', 'arrhythmia', 'bank', 'compasmen', 'compaswomen', 'contraceptive', 
            'german', 'heart', 'student', 'titanic']
# Set - classification algorithms
models = ['ad', 'dt', 'knn', 'mlp', 'rf', 'svm', 'xgb']

def report_results():
    measures = ['f1_macro', 'ratio_f1_macro', 'ratio_selection_rate', 'ratio_recall', 'ratio_odds']
    df = pd.DataFrame(columns = ['Dataset', 'Config', 'Selection'] + measures)
    count = 0

    for name in datasets:    
        for config in models:
            if os.path.exists(path + name + '_' + config + '_validation.csv'):
                validation = pd.read_csv(path + name + '_' + config + '_validation.csv', 
                                         sep=';', index_col=0)
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                validation['AHP-G.'] = list(GaussianAHP(criteria).calculate(validation))
                validation['MCPM'] = list(MCPM(criteria).calculate(validation))
                validation['SAC'] = list(SAC(criteria).calculate(validation))

                test = pd.read_csv(path + name + '_' + config + '_test.csv', sep=';', index_col=0)
                test['ratio_odds'] = (test['ratio_fpr'] + test['ratio_recall'])/2 

                idx_AHP = validation['AHP-G.'].idxmax()
                idx_MCPM = validation['MCPM'].idxmax()
                idx_SAC = validation['SAC'].idxmax()

                ahp = []
                mcpm = []
                sac = []

                ahp.append(name)
                mcpm.append(name)
                sac.append(name)

                ahp.append(config)
                mcpm.append(config)
                sac.append(config)

                ahp.append('AHP-G.')
                mcpm.append('MCPM')
                sac.append('SAC')
                
                for measure in measures:
                    ahp.append(test.loc[idx_AHP, measure])
                    mcpm.append(test.loc[idx_MCPM, measure])
                    sac.append(test.loc[idx_SAC, measure])
                    
                df.loc[count] = ahp 
                count += 1
                df.loc[count] = mcpm 
                count += 1
                df.loc[count] = sac
                count += 1
                
    return df

def statistical(measure):

    df_statistical = pd.DataFrame(columns = ['AHP-G.', 'MCPM', 'SAC'])

    count = 0

    for name in datasets:    
        for config in models:
            if os.path.exists(path + name + '_' + config + '_validation.csv'):
                validation = pd.read_csv(path + name + '_' + config + '_validation.csv', sep=';', index_col=0)
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
                validation['AHP-G.'] = list(GaussianAHP(criteria).calculate(validation))
                validation['MCPM'] = list(MCPM(criteria).calculate(validation))
                validation['SAC'] = list(SAC(criteria).calculate(validation))

                test = pd.read_csv(path + name + '_' + config + '_test.csv', sep=';', index_col=0)
                test['ratio_odds'] = (test['ratio_fpr'] + test['ratio_recall'])/2

                idx_AHP = validation['AHP-G.'].idxmax()
                idx_MCPM = validation['MCPM'].idxmax()
                idx_SAC = validation['SAC'].idxmax()

                df_statistical.loc[count] = [test.loc[idx_AHP, measure], test.loc[idx_MCPM, measure],
                                      test.loc[idx_SAC, measure]]
                count += 1
    return df_statistical

def cd(results, name):
    df = results[['AHP-G.', 'MCPM', 'SAC']]
    algorithms_names = ['AHP-G.', 'MCPM', 'SAC']
    results_array = df.values
    ranks_test = np.array([rankdata(-p) for p in results_array])
    average_ranks = np.mean(ranks_test, axis=0)
    print('\n'.join('({}) Rank Average: {}'.format(a, r) for a, r in zip(algorithms_names, average_ranks)))
    cd = compute_CD(average_ranks, n=len(df), alpha='0.05', test='nemenyi')
    print('CD = ', cd)
    graph_ranks(average_ranks, names=algorithms_names, cd=cd, width=3.5, textspace=0.85, reverse=False, 
                filename = name + '.pdf')
    


