import os
import logging
import numpy as np
from gensim.models import KeyedVectors
import urllib.request
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union, Dict
import pickle

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

# === NEW SENTENCE TRANSFORMER FUNCTIONS ===

class EnhancedEmbedder:
    """
    Enhanced embedder that supports both FastText and Sentence Transformers
    """
    
    def __init__(self, embedding_type='fasttext', model_name_or_path=None):
        self.embedding_type = embedding_type
        self.model = None
        self.vector_size = None
        
        if embedding_type == 'fasttext':
            if model_name_or_path:
                self.model = load_gensim_vec(model_name_or_path)
                self.vector_size = self.model.vector_size
        elif embedding_type == 'sentence_transformer':
            model_name = model_name_or_path or 'all-MiniLM-L6-v2'
            logger.info(f"Loading sentence transformer: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # Get embedding dimension by encoding a sample text
            sample_embedding = self.model.encode("sample text")
            self.vector_size = len(sample_embedding)
            logger.info(f"Sentence transformer loaded with embedding dimension: {self.vector_size}")
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embeddings based on the configured embedding type
        """
        if self.embedding_type == 'fasttext':
            return self._fasttext_encode(text)
        elif self.embedding_type == 'sentence_transformer':
            return self._sentence_transformer_encode(text)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        NEED RECHECK
        Encode multiple texts into embeddings
        """
        if self.embedding_type == 'sentence_transformer':
            # Sentence transformers can batch encode efficiently
            return self.model.encode(texts)
        else:
            # For FastText, encode individually
            return np.array([self.encode_text(text) for text in texts])
    
    def _fasttext_encode(self, text: str) -> np.ndarray:
        """
        FastText encoding - average word embeddings
        """
        if '|' in text:
            words = text.split('|')
        else:
            words = text.split()
        
        words = [w.strip() for w in words if w.strip()]
        vectors = [self.model[w] for w in words if w in self.model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)
    
    def _sentence_transformer_encode(self, text: str) -> np.ndarray:
        """
        Sentence transformer encoding - uses the full sentence context
        """
        # Handle pipe-separated keywords by joining them into sentences
        if '|' in text:
            # Convert pipe-separated keywords to natural sentence
            keywords = text.split('|')
            keywords = [w.strip() for w in keywords if w.strip()]
            text = ' '.join(keywords)
        
        return self.model.encode(text)
    
    def save_model(self, path: str):
        """
        Save the model for later use
        """
        if self.embedding_type == 'sentence_transformer':
            self.model.save(path)
            logger.info(f"Sentence transformer saved to {path}")
        else:
            logger.info("FastText model saving not implemented (use original file)")
    
    def load_model(self, path: str):
        """
        Load a saved model
        """
        if self.embedding_type == 'sentence_transformer':
            self.model = SentenceTransformer(path)
            logger.info(f"Sentence transformer loaded from {path}")

def create_embedder(embedding_type: str, model_path: str = None) -> EnhancedEmbedder:
    """
    Factory function to create an embedder based on configuration
    """
    return EnhancedEmbedder(embedding_type=embedding_type, model_name_or_path=model_path)

def process_firm_keywords_enhanced(keywords_text: str, embedder: EnhancedEmbedder) -> np.ndarray:
    """
    Process firm keywords using enhanced embedder
    """
    return embedder.encode_text(keywords_text)

def process_patent_abstracts_enhanced(abstracts: List[str], embedder: EnhancedEmbedder) -> np.ndarray:
    """
    Process patent abstracts using enhanced embedder
    """
    if len(abstracts) == 1:
        return embedder.encode_text(abstracts[0])
    
    # For multiple abstracts, we can either:
    # 1. Average their embeddings
    # 2. Concatenate and encode as one text
    # 3. Use each separately
    
    # Option 1: Average embeddings (maintains compatibility)
    embeddings = embedder.encode_texts(abstracts)
    return np.mean(embeddings, axis=0)

def process_user_query_enhanced(query: str, embedder: EnhancedEmbedder) -> np.ndarray:
    """
    Process user query using enhanced embedder
    """
    return embedder.encode_text(query)

# === RAG UTILITIES ===

def create_rag_documents(company_keywords_df, use_external_summaries=False, external_summaries_path=None):
    """
    Create RAG documents from company keywords or external summaries
    """
    documents = []
    
    if use_external_summaries and external_summaries_path and os.path.exists(external_summaries_path):
        logger.info("Using external company summaries for RAG")
        import pandas as pd
        summaries_df = pd.read_csv(external_summaries_path)
        
        for _, row in summaries_df.iterrows():
            documents.append({
                'company_id': str(row['company_id']),
                'company_name': row.get('company_name', 'Unknown'),
                'content': row['summary'],
                'metadata': {
                    'source': 'external_summary',
                    'company_id': str(row['company_id'])
                }
            })
    else:
        logger.info("Using dual attention keywords for RAG")
        
        # Debug: Check original dataframe columns
        logger.info(f"Original dataframe columns: {list(company_keywords_df.columns)}")
        logger.info(f"Original dataframe shape: {company_keywords_df.shape}")
        
        # Load company names if not already in the dataframe
        if 'company_name' not in company_keywords_df.columns:
            try:
                from configs.paths import US_WEB_DATA
                import pandas as pd
                logger.info("Loading company names from web data...")
                
                # Check if US_WEB_DATA file exists
                if not os.path.exists(US_WEB_DATA):
                    logger.error(f"US_WEB_DATA file not found: {US_WEB_DATA}")
                    company_keywords_df['company_name'] = 'Unknown'
                else:
                    us_web_data = pd.read_csv(US_WEB_DATA)
                    logger.info(f"Loaded US web data with shape: {us_web_data.shape}")
                    logger.info(f"US web data columns: {list(us_web_data.columns)}")
                    
                    company_name_map = us_web_data[['hojin_id', 'company_name']].drop_duplicates()
                    logger.info(f"Company name map shape: {company_name_map.shape}")
                    
                    # Convert hojin_id to same type for merging
                    company_keywords_df['hojin_id'] = company_keywords_df['hojin_id'].astype(str)
                    company_name_map['hojin_id'] = company_name_map['hojin_id'].astype(str)
                    
                    # Check a few sample IDs for debugging
                    sample_ids = company_keywords_df['hojin_id'].head(5).tolist()
                    logger.info(f"Sample company IDs from keywords: {sample_ids}")
                    matching_names = company_name_map[company_name_map['hojin_id'].isin(sample_ids)]
                    logger.info(f"Matching names found: {len(matching_names)}")
                    
                    # Merge to get company names
                    before_merge_shape = company_keywords_df.shape
                    company_keywords_df = company_keywords_df.merge(company_name_map, on='hojin_id', how='left')
                    after_merge_shape = company_keywords_df.shape
                    
                    logger.info(f"Merge completed: {before_merge_shape} -> {after_merge_shape}")
                    
                    # Check how many company names we got
                    non_null_names = company_keywords_df['company_name'].notna().sum()
                    logger.info(f"Successfully merged company names for {non_null_names}/{len(company_keywords_df)} companies")
                    
                    # Fill any remaining NaN values with 'Unknown'
                    company_keywords_df['company_name'] = company_keywords_df['company_name'].fillna('Unknown')
                    
            except Exception as e:
                logger.error(f"Could not load company names: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Add a default company_name column
                company_keywords_df['company_name'] = 'Unknown'
        else:
            logger.info("Company names already present in dataframe")
        
        # Debug: Check final dataframe columns
        logger.info(f"Final dataframe columns: {list(company_keywords_df.columns)}")
        
        for _, row in company_keywords_df.iterrows():
            # Convert keywords to more readable format
            keywords = row['company_keywords'].split('|') if '|' in str(row['company_keywords']) else [str(row['company_keywords'])]
            keywords = [k.strip() for k in keywords if k.strip()]
            
            content = f"This company specializes in: {', '.join(keywords)}"
            
            documents.append({
                'company_id': str(row['hojin_id']),
                'company_name': row.get('company_name', 'Unknown'),
                'content': content,
                'metadata': {
                    'source': 'dual_attention_keywords',
                    'company_id': str(row['hojin_id']),
                    'keywords': keywords
                }
            })
    
    logger.info(f"Created {len(documents)} RAG documents")
    
    # Debug: Check a few sample documents
    if documents:
        logger.info("Sample document company names:")
        for i, doc in enumerate(documents[:5]):
            logger.info(f"  {i+1}. ID: {doc['company_id']}, Name: {doc['company_name']}")
    
    return documents

def save_embeddings_cache(embeddings_dict: Dict, cache_path: str):
    """
    Save embeddings to cache file
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    logger.info(f"Saved embeddings cache to {cache_path}")

def load_embeddings_cache(cache_path: str) -> Dict:
    """
    Load embeddings from cache file
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        logger.info(f"Loaded embeddings cache from {cache_path}")
        return embeddings_dict
    return {}

# Backward compatibility functions
def text_to_vector_enhanced(text, embedder):
    """
    Enhanced version of text_to_vector that works with new embedder
    """
    return embedder.encode_text(text)



