from ast import expr
from itertools import count
from collections import Counter
import numpy as np
import nltk
from nltk import CFG
from nltk.parse.generate import generate
import spacy

TEXT_SANDBOX_GRAMMAR = """
S -> 'ite' B L
B -> 'count' TOK OP INT | 'contains' TOK STR
TOK -> '<br' | 'Follow' | 'check' | 'channel' | 'plz' | 'PSY' | 'like' | 'song' | 'video' | '!!!!'
OP -> '>' | '<' | '='
INT -> '0' | '1' | '2' | '3' | '4' | '5'
STR -> '!' | '&' | '*' | '@'
L -> '(1, -1)' | '(-1, 1)'
"""

'''
Boilerplate for the labeling function synthesis engine.
'''
class TextLabelFunction:

    def __init__(self, expression) -> None:
        self.expression = expression
        self.string_expr = ' '.join(expression)
    
    def generate_synthesized_function(self, sentence):
        if self.expression[0] == 'ite': # go to if-then-else logic
            # get the expr tokens and the labelset manually
            # TODO: change this when the grammar gets more complicated / recursive
            return self.evaluate_ite(self.expression[1: -1], self.expression[-1], sentence)
    
    def evaluate_contains(self, token, checkstring, sentence):
        if checkstring in token and token in sentence:
            return True
        return False
    
    def evaluate_count(self, token, op, int_num, sentence):
        cntr = Counter(sentence)
        if op == '<':
            return cntr[token] < int(int_num)
        elif op == '>':
            return cntr[token] > int(int_num)
        else:
            return cntr[token] == int(int_num)
    
    def evaluate_ite(self, expres, labelset, sentence):
        label_tuple = eval(labelset)
        if expres[0] == 'count':
            result = self.evaluate_count(expres[1], expres[2], expres[3], sentence)
        else: # expression is contains. TODO: update this when we add more expressions
            assert expres[0] == 'contains'
            result = self.evaluate_contains(expres[1], expres[2], sentence)
        return label_tuple[0] if result else label_tuple[1]

def enumerate_from_grammar(grammar_str, function_class):
    cfg_gnr = CFG.fromstring(grammar_str)
    for expr in generate(cfg_gnr):
        LF = function_class(expr)
        yield LF