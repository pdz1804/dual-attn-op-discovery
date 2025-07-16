import os
import logging
import numpy as np
from gensim.models import KeyedVectors
import urllib.request

logger = logging.getLogger(__name__)

def download_aligned_vec(lang, target_dir):
    filename = f'wiki.{lang}.align.vec'
    out_path = os.path.join(target_dir, filename)
    if os.path.exists(out_path):
        logger.info(f"{filename} already exists.")
        return out_path
    url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/{filename}'
    urllib.request.urlretrieve(url, out_path)
    logger.info(f"Downloaded {lang} vectors to {out_path}")
    return out_path

def load_gensim_vec(path):
    logger.info(f"Loading FastText vectors from {path}")
    return KeyedVectors.load_word2vec_format(path, binary=False)

def text_to_vector(text, ft_model):
    words = text.split('|')
    words = [w.strip() for w in words if w.strip()]
    vectors = [ft_model[w] for w in words if w in ft_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(ft_model.vector_size)


