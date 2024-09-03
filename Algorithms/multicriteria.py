import numpy as np
import pandas as pd
import math

from itertools import combinations


class Multicriteria:
    def __init__(self, criteria_list, inverted_criteria = []):
        self.criteria = criteria_list
        self.inverted_criteria = inverted_criteria
        self.results = None
    
    def ranking(self):
        return self.results.sort_values(by='value', ascending=False).reset_index(drop=True)
    
class SAC(Multicriteria):
    def __init__(self, criteria_list, inverted_criteria = []):
        super().__init__(criteria_list, inverted_criteria)
        
    def calculate(self, aux_decision_matrix):
        decision_matrix = aux_decision_matrix[self.criteria].copy()
        self.results = pd.DataFrame(columns=['item', 'value'])
        decision_matrix[self.inverted_criteria] = 1 - decision_matrix[self.inverted_criteria]
        self.results['item'] = decision_matrix.index
        self.results['value'] = list(decision_matrix.sum(axis=1))
        return self.results['value']

class MCPM(Multicriteria):
    def __init__(self, criteria_list, inverted_criteria = []):
        super().__init__(criteria_list, inverted_criteria)
        
    def multicriteria(self, values):
        pairs = list(combinations(values, 2))
        area = 0
        for a, b in pairs:
            area += (a*b*math.sin((2*math.pi)/3)/2)    
        return area

    def calculate(self, aux_decision_matrix):
        decision_matrix = aux_decision_matrix[self.criteria].copy()
        self.results = pd.DataFrame(columns=['item', 'value'])
        decision_matrix[self.inverted_criteria] = 1 - decision_matrix[self.inverted_criteria]
        matrix = decision_matrix[self.criteria].to_numpy()
        self.results['item'] = decision_matrix.index
        self.results['value'] = [self.multicriteria(row) for row in matrix]
        return self.results['value']
                
class GaussianAHP(Multicriteria):
    def __init__(self, criteria_list, inverted_criteria = []):
        super().__init__(criteria_list, inverted_criteria)
        
    def calculate(self, aux_decision_matrix):
        decision_matrix = aux_decision_matrix[self.criteria].copy()
        self.results = pd.DataFrame(columns=['item', 'value'])
        decision_matrix[self.inverted_criteria] = 1 - decision_matrix[self.inverted_criteria]
        decision_matrix = decision_matrix/decision_matrix.sum()
        ''' Normalized Gaussian Factor '''
        ngf = (decision_matrix.std()/decision_matrix.mean())/(decision_matrix.std()/decision_matrix.mean()).sum()
        decision_matrix = decision_matrix*ngf
        self.results['item'] = decision_matrix.index
        self.results['value'] = list(decision_matrix.sum(axis=1))
        return self.results['value']
        