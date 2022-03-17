import wrench.wrench as wrnch
import spacy
import numpy as np
import nltk

import logging
import torch

from wrench.wrench.dataset import load_dataset
from wrench.wrench.logging import LoggingHandler
from wrench.wrench.endmodel import MLPModel
from wrench.wrench.labelmodel import MajorityVoting, FlyingSquid
from typing import Any, List, Optional, Union, Callable


class LabelFunctionSet:
    
    def __init__(self, training_data: Optional[List]=None, validation_data:Optional[List]=None, 
                 initial_functions: Optional[List]=None):
        self.train_data = training_data
        self.valid_data = validation_data
        self._labeling_functions = [] if initial_functions is None else initial_functions
    
    def add_function(self, labeling_function: Callable):
        self._labeling_functions.append(labeling_function)
    
    def reset_functions(self):
        self._labeling_functions = []

    def apply_labels(self, dataset):
        #TODO: speed this up and vectorize, if possible
        label_array = []
        assert len(self._labeling_functions) != 0, "Need to add labeling functions in order to apply them!"
        for example in dataset.examples:
            example_labels = []
            for lf in self._labeling_functions:
                example_labels.append(lf(example['text']))
            label_array.append(example_labels)
        # convert into a numpy.ndarray for compatibility with WRENCH
        label_array = np.array(label_array)
        return label_array

class RandomLF:

    def __init__(self, output_size: int=2):
        self.output_size = output_size
    
    def apply(self, datapoint):
        return np.random.randint(self.output_size)

class RandomLFGenerator:

    def __init__(self, num_functions: int, output_size: int=2) -> None:
        self.fxns = []
        for fxnidx in range(num_functions):
            self.fxns.append(RandomLF(output_size=output_size))
    
    def get_random_lfs(self):
        #TODO: there is likely a better way of doing this
        fxnmethods = []
        for fxn in self.fxns:
            fxnmethods.append(fxn.apply)
        return fxnmethods
