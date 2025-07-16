#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 00:27:59 2022
@author: zhuchen
"""

from __future__ import division
import numpy as np

class Tokenizer(object):
    def __init__(self, words, data, max_len = 256):
        num_words = np.array(range(len(words))) + 1
        self.data = data
        self.label_df = data[['hojin_id', 'hightechflag']].drop_duplicates()
        self.max_len = max_len
        self.token_to_id = dict(zip(words, num_words))
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
    def encode(self, text):
        input_ids = [self.token_to_id[w] for w in text if w in self.token_to_id]
        sen_len = len(input_ids)
        if sen_len > self.max_len:
            return input_ids[:self.max_len]
        else:
            return input_ids + [0] * (self.max_len - sen_len)
        
    def encode_webportfolio(self, company_id, max_page = 32):
        web_portfolio = list(self.data[self.data.hojin_id == company_id].cleaned_content)
        num_page = len(web_portfolio)
        web_vectors = [self.encode(text.split('|')) for text in web_portfolio]
        if num_page >= max_page:
            return max_page, web_vectors
        elif num_page < max_page: # padding web page
            return num_page, web_vectors + [[0] * self.max_len for i in range(max_page-num_page)]
        
    def get_label(self, company_id):
        return int(list(self.label_df[self.label_df.hojin_id == company_id].hightechflag)[0])


