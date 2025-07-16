# Import necessary libraries
from __future__ import division
import logging
import sys

# === Create a fresh log file every time ===
LOG_FILE_PATH = "pdzttb.log"

# Remove any existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up new logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),  # Overwrite each run
        logging.StreamHandler(sys.stdout),  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os 
import torch
import torch.nn as nn
from tqdm import tqdm
import ast  # To safely evaluate string list to real list
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import fasttext.util
import gzip
import shutil
import random 
import time
import pickle
import urllib.request
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import KeyedVectors
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Import from files 
from data_loader.web_data_preprocess import clean_tokens
from data_loader.tokenizer import Tokenizer
from models.layers.fast_vector import FastVector
from models.dual_attention import DualAttnModel
from models.patent2product import Patent2Product # MODEL A: PATENT → PRODUCT
from models.product2patent import Product2Patent # MODEL B: PRODUCT → PATENT
from configs.paths import *
from configs.hyperparams import *
from utils.count_params import count_parameters
from utils.colorize_attention import colorize
from utils.select_keywords import select_keywords
from utils.plot_utils import plot_loss, plot_accuracy
from utils.seed_everything import set_seed
from training.train_dual_attention import train
from training.evaluate import evaluate
from inference.query_opportunity_best import query_opportunity_product_best
from inference.query_opportunity_matrix import query_opportunity_product_matrix_only

import argparse
from pipelines import dual_attention_pipeline, patent_product_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, choices=['dual_attn', 'patent_product'], required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'chat'], default=None, help="Train/Test/Chat mode (only for patent_product pipeline)")

    args = parser.parse_args()
    
    set_seed(42)  # Set seed for reproducibility

    if args.pipeline == 'dual_attn':
        dual_attention_pipeline.run()

    elif args.pipeline == 'patent_product':
        if args.mode is None:
            raise ValueError("Error: --mode is required for patent_product pipeline. Use --mode=train / test / chat.")

        if args.mode == 'train':
            patent_product_pipeline.train_pipeline()
        elif args.mode == 'test':
            patent_product_pipeline.test_pipeline()
        elif args.mode == 'chat':
            patent_product_pipeline.test_pipeline_chat()

if __name__ == "__main__":
    # your pipeline code here
    main()
    exit(1)  # Exit after running the pipeline
    
    # ============================================
    # Transformation Matrix Training 
    # Download aligned vectors 

    def download_aligned_vec(lang, target_dir='.'):
        assert lang in ('en', 'zh')
        os.makedirs(target_dir, exist_ok=True)

        filename = f'wiki.{lang}.align.vec'
        out_path = os.path.join(target_dir, filename)

        # Skip download if file already exists
        if os.path.exists(out_path):
            logger.info(f'{filename} already exists. Skipping download.')
            return out_path

        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/{filename}'
        
        logger.info(f'Downloading {lang} aligned vector: {url}')
        urllib.request.urlretrieve(url, out_path)
        logger.info(f'Saved to {out_path}')
        return out_path

    lst = ['en']

    for lg in lst:
        download_aligned_vec(lg, target_dir=DATA_DIR)
        # download_aligned_vec(lg, target_dir=os.getcwd())    
        
    # load patent data with keyword data for 3 countries 

    data_info = {
        'CN': {
            'patent_file': '/content/CN_JP_US_data/cn_patent.csv',
            'fasttext_model': '/content/CN_JP_US_data/wiki.zh.align.vec',
            'output_model': '/content/CN_JP_US_data/patent2product_cn.pt'
        },
        'JP': {
            'patent_file': '/content/CN_JP_US_data/jp_patent.csv',
            'fasttext_model': '/content/CN_JP_US_data/vectors-ja.txt',
            'output_model': '/content/CN_JP_US_data/patent2product_jp.pt'
        },
        # 'US': {
        #     'patent_file': '/content/CN_JP_US_data/us_patent.csv',
        #     'fasttext_model': '/content/CN_JP_US_data/wiki.en.align.vec',
        #     'output_model': '/content/CN_JP_US_data/patent2product_us.pt'
        # }
        'US': {
            # 'patent_file': './CN_JP_US_data/us_patent202506.csv',
            'patent_file': './CN_JP_US_data/cleaned_patents.csv',
            # 'fasttext_model': './CN_JP_US_data/wiki.en.align.vec',
            'fasttext_model': 'fasttext_web_patent.vec',
            'output_model': './CN_JP_US_data/patent2product_us.pt'
        }
    }

    # Output from the Dual Attention model
    our_dual_att_result = pd.read_csv(
        EMBEDDINGS_OUTPUT,
        index_col=0
    )

    data_path = US_WEB_DATA
    us_web_with_company = pd.read_csv(data_path)

    # Keep only necessary columns from `us_web_with_company`
    company_name_map = us_web_with_company[['hojin_id', 'company_name']].drop_duplicates()

    # Merge based on hojin_id
    merged_df = our_dual_att_result.merge(company_name_map, on='hojin_id', how='left')

    if 'Product embedding vector' not in merged_df.columns:
        merged_df.insert(merged_df.columns.get_loc('company_keywords') + 1, 'Product embedding vector', '')
        
    merged_df['Product embedding vector'] = ''

    # Rename columns
    merged_df = merged_df.rename(columns={
        'hojin_id': 'Firm ID',
        'company_name': 'Firm name',
        'company_keywords': 'Keywords list',
        'embedding': 'Embedding'
    })

    merged_df['Firm country'] = 'US'
    keyword_df = merged_df

    # Ensure Firm ID is string (for consistency with other dicts)
    keyword_df['Firm ID'] = keyword_df['Firm ID'].astype(str)

    # Create the mapping: Firm ID → Firm name
    firm_id_name_map = dict(zip(keyword_df['Firm ID'], keyword_df['Firm name']))
    
    # LOAD DATA AND CONVERT TO VECTORS 
    def load_gensim_vec(path):
        logger.info(f"Loading vectors: {path}")
        return KeyedVectors.load_word2vec_format(path, binary=False)

    def text_to_vector(text, ft_model, top_k=3, debug=False):
        words = text.split('|')
        words = [w.strip() for w in words if w.strip()]
        vectors = [ft_model[w] for w in words if w in ft_model]

        if vectors:
            return np.mean(vectors, axis=0)

        return np.zeros(ft_model.vector_size)

    patent_rep_dict = {}
    product_rep_dict = {}
    shared_ids = {}
    firm_patent_ids = {}  # NEW: Map firm_id → list of appln_ids
    patent_text_map = {}  # NEW: Map appln_id → cleaned abstract text

    for country, info in data_info.items():
        logger.info(f"Processing: {country}")
        
        if country in ['CN', 'JP']:
            continue  # Skip CN and JP for now

        patent_df = pd.read_csv(info['patent_file'])
        logger.info(f"  Loaded patent file with {len(patent_df)} rows.")

        ft_model = load_gensim_vec(info['fasttext_model'])
        logger.info(f"  Loaded FastText model for {country}.")

        country_indices = keyword_df[keyword_df['Firm country'] == country].index
        for idx in country_indices:
            firm_id = str(keyword_df.at[idx, 'Firm ID'])
            keywords = keyword_df.at[idx, 'Keywords list']
            product_vec = text_to_vector(keywords, ft_model, debug=True)
            product_rep_dict[firm_id] = product_vec

        abstract_col = next((col for col in patent_df.columns if 'clean_abstract' in col.lower()), None)
        if abstract_col is None:
            raise ValueError(f"Can't find {country} patent abstracts")

        logger.info(f"  Using abstract column: {abstract_col}")

        firm_counter = 0  # Limit debug prints to 2 firms
        for firm_id, group in patent_df.groupby('hojin_id'):
            firm_id = str(firm_id)

            raw_lists = group[abstract_col].dropna().tolist()
            appln_ids = group['appln_id'].tolist()
            
            abstracts_with_ids = group[[abstract_col, 'appln_id']].dropna().values.tolist()
            # abstracts_with_ids = group[['patent_abstract', 'appln_id']].dropna().values.tolist()
            
            all_keywords = []

            # New
            for abstract, appln_id in abstracts_with_ids:
                if isinstance(abstract, str):
                    try:
                        tokens = eval(abstract) if abstract.startswith('[') else abstract.split()
                    except:
                        tokens = abstract.split()
                else:
                    tokens = abstract

                all_keywords.extend(tokens)
                patent_text_map[appln_id] = ' '.join(tokens[:20])

            if firm_counter < 2:
                logger.info(f"    Total tokens collected: {len(all_keywords)}")
                logger.info(f"    Sample tokens: {all_keywords[:20]}")
                logger.info(f"    Final joined text preview: {'|'.join(all_keywords[:50])}...")

            all_text = '|'.join(all_keywords)
            patent_vec = text_to_vector(all_text, ft_model, debug=True)

            if firm_counter < 2:
                logger.info(f"    Patent vector norm: {np.linalg.norm(patent_vec):.4f}")

            patent_rep_dict[firm_id] = patent_vec
            firm_patent_ids[firm_id] = appln_ids  # Track associated patents

            firm_counter += 1

        shared_ids_local = list(set(product_rep_dict.keys()) & set(patent_rep_dict.keys()))
        shared_ids[country] = shared_ids_local
        
        logger.info(f"  Matched {len(shared_ids[country])} firms in {country} with both patent and product data.")

        if len(shared_ids[country]) == 0:
            logger.info(f"  ⚠️ Skip {country}: no matched data")
            continue

    patent_df = pd.read_csv(data_info["US"]['patent_file'])

    # PREPARE TRAINING DATA 
    models_A = {}
    models_B = {}

    losses_A = {}
    losses_B = {}

    transformation_matrices_A = {}
    transformation_matrices_B = {}

    EPOCH = EPOCHS_TRANS_MATRIX

    # MODEL A: PATENT → PRODUCT
        
    for country, info in data_info.items():
        if country in ['CN', 'JP'] or len(shared_ids[country]) == 0:
            continue

        logger.info(f"[Train A] Processing: {country}")
        
        # Prepare data
        X_all = np.stack([patent_rep_dict[i] for i in shared_ids[country]])
        Y_all = np.stack([product_rep_dict[i] for i in shared_ids[country]])

        X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32)

        # Model & Optimizer
        model_A = Patent2Product()
        optimizer = optim.Adam(model_A.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        train_loss_A, val_loss_A = [], []
        train_cos_A, val_cos_A = [], []

        for epoch in range(EPOCH):
            # ----- Train -----
            model_A.train()
            optimizer.zero_grad()
            pred = model_A(X_train)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            optimizer.step()
            train_loss_A.append(loss.item())

            # Track train cosine similarity
            cos_train = torch.nn.functional.cosine_similarity(pred, Y_train, dim=1).mean().item()
            train_cos_A.append(cos_train)

            # ----- Validation -----
            model_A.eval()
            with torch.no_grad():
                val_pred = model_A(X_val)
                val_loss = loss_fn(val_pred, Y_val)
                val_loss_A.append(val_loss.item())

                # Track val cosine similarity
                cos_val = torch.nn.functional.cosine_similarity(val_pred, Y_val, dim=1).mean().item()
                val_cos_A.append(cos_val)

            logger.info(f"[A-{country}] Epoch {epoch}: "
                f"Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, "
                f"Train Cos = {cos_train:.4f}, Val Cos = {cos_val:.4f}")

        # Save model and loss
        models_A[country] = model_A
        losses_A[country] = {
            'train': train_loss_A,
            'val': val_loss_A,
            'train_cos': train_cos_A,
            'val_cos': val_cos_A
        }

        # Save transformation matrix A = W2 @ W1
        W1 = model_A.net[0].weight.detach().cpu().numpy()
        W2 = model_A.net[2].weight.detach().cpu().numpy()
        A = W2 @ W1
        transformation_matrices_A[country] = A

    for country in models_A:
        logger.info(f"[Evaluation A] {country}")

        # Plot loss
        plt.figure(figsize=(8, 4))
        plt.plot(losses_A[country]['train'], label='Train Loss')
        plt.plot(losses_A[country]['val'], label='Val Loss')
        plt.title(f'Patent2Product Loss - {country}')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot cosine similarity
        plt.figure(figsize=(8, 4))
        plt.plot(losses_A[country]['train_cos'], label='Train Cosine Similarity')
        plt.plot(losses_A[country]['val_cos'], label='Val Cosine Similarity')
        plt.title(f'Patent2Product Cosine Similarity - {country}')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Final full cosine evaluation (on all data)
        model = models_A[country]
        model.eval()
        X_all = np.stack([patent_rep_dict[i] for i in shared_ids[country]])
        Y_all = np.stack([product_rep_dict[i] for i in shared_ids[country]])

        with torch.no_grad():
            pred_all = model(torch.tensor(X_all, dtype=torch.float32))
            cos_sims = torch.nn.functional.cosine_similarity(
                pred_all, torch.tensor(Y_all, dtype=torch.float32), dim=1
            ).numpy()
            
            logger.info(f"[A-{country}] Mean Cosine similarity: {cos_sims.mean():.4f}, std: {cos_sims.std():.4f}")

    # MODEL B: PRODUCT → PATENT
    for country, info in data_info.items():
        if country in ['CN', 'JP'] or len(shared_ids[country]) == 0:
            continue

        logger.info(f"[Train B] Processing: {country}")
        
        # Prepare reversed data: X = Product, Y = Patent
        X_all = np.stack([product_rep_dict[i] for i in shared_ids[country]])
        Y_all = np.stack([patent_rep_dict[i] for i in shared_ids[country]])

        X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32)

        # Model & Optimizer
        model_B = Product2Patent()
        optimizer = optim.Adam(model_B.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        train_loss_B, val_loss_B = [], []
        train_cos_B, val_cos_B = [], []  # Track cosine similarity

        for epoch in range(EPOCH):
            model_B.train()
            optimizer.zero_grad()
            pred = model_B(X_train)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            optimizer.step()
            train_loss_B.append(loss.item())

            # Cosine sim (train)
            cos_train = torch.nn.functional.cosine_similarity(pred, Y_train, dim=1).mean().item()
            train_cos_B.append(cos_train)

            # Validation
            model_B.eval()
            with torch.no_grad():
                val_pred = model_B(X_val)
                val_loss = loss_fn(val_pred, Y_val)
                val_loss_B.append(val_loss.item())

                cos_val = torch.nn.functional.cosine_similarity(val_pred, Y_val, dim=1).mean().item()
                val_cos_B.append(cos_val)

            logger.info(f"[B-{country}] Epoch {epoch}: "
                f"Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, "
                f"Train Cos = {cos_train:.4f}, Val Cos = {cos_val:.4f}")

        # Save model and loss
        models_B[country] = model_B
        losses_B[country] = {
            'train': train_loss_B,
            'val': val_loss_B,
            'train_cos': train_cos_B,
            'val_cos': val_cos_B
        }

        # Save transformation matrix B = W2 @ W1
        W1 = model_B.net[0].weight.detach().cpu().numpy()
        W2 = model_B.net[2].weight.detach().cpu().numpy()
        B = W2 @ W1
        transformation_matrices_B[country] = B

    for country in models_B:
        logger.info(f"[B-{country}] Model evaluation")

        # Plot loss
        plt.figure(figsize=(8, 4))
        plt.plot(losses_B[country]['train'], label='Train Loss')
        plt.plot(losses_B[country]['val'], label='Val Loss')
        plt.title(f'Product2Patent Loss - {country}')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot cosine similarity
        plt.figure(figsize=(8, 4))
        plt.plot(losses_B[country]['train_cos'], label='Train Cosine Similarity')
        plt.plot(losses_B[country]['val_cos'], label='Val Cosine Similarity')
        plt.title(f'Product2Patent Cosine Similarity - {country}')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Cosine stats on all data
        model = models_B[country]
        model.eval()

        X_all = np.stack([product_rep_dict[i] for i in shared_ids[country]])
        Y_all = np.stack([patent_rep_dict[i] for i in shared_ids[country]])

        with torch.no_grad():
            pred_all = model(torch.tensor(X_all, dtype=torch.float32))
            cos_sims = torch.nn.functional.cosine_similarity(
                pred_all, torch.tensor(Y_all, dtype=torch.float32), dim=1
            ).numpy()
            
            logger.info(f"[B-{country}] Mean Cosine similarity: {cos_sims.mean():.4f}, std: {cos_sims.std():.4f}")

    # SAVE MODEL AND APPLY TO FULL DATASET 
    us_firms = keyword_df[keyword_df['Firm country'] == 'US']
    missing_in_patent_dict = us_firms[~us_firms['Firm ID'].astype(str).isin(patent_rep_dict.keys())]

    logger.info(f"Total US firms: {len(us_firms)}")
    logger.info(f"Missing US firms in patent_rep_dict: {len(missing_in_patent_dict)}")

    output_path = FIRM_PRODUCT_KEYWORDS

    # Ensure the new column exists first
    if 'Product embedding vector' not in keyword_df.columns:
        keyword_df['Product embedding vector'] = ""

    for country, info in data_info.items():
        if country in ['CN', 'JP']:
            continue  # Skip these countries

        logger.info(f"[PROCESS] {country} — Generating product vectors from patent vectors.")

        model = models_B[country]
        model.eval()

        # Get all indices from keyword_df where the firm is from the current country
        country_indices = keyword_df[keyword_df['Firm country'] == country].index

        for idx in country_indices:
            firm_id = str(keyword_df.at[idx, 'Firm ID'])

            if firm_id in patent_rep_dict:
                x_vec = torch.tensor(patent_rep_dict[firm_id], dtype=torch.float32).unsqueeze(0)  # shape: [1, D]
                with torch.no_grad():
                    y_pred = model(x_vec).squeeze().numpy()  # shape: [output_dim]
                vector_str = '|'.join(map(str, y_pred))
                keyword_df.at[idx, 'Product embedding vector'] = vector_str

    logger.info(f"[SAVE] Writing output to: {output_path}")
    keyword_df.to_csv(output_path, index=False)

    test = pd.read_csv(output_path)

    # EXPORT TRANSFORMATION MATRICES
    save_dir = MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    for country in models_A:
        # Save model A
        path_A = os.path.join(save_dir, f"{country}_Patent2Product.pt")
        torch.save(models_A[country].state_dict(), path_A)

        # Save matrix A
        matrix_A_path = path_A.replace('.pt', '_transform.npy')
        np.save(matrix_A_path, transformation_matrices_A[country])

    for country in models_B:
        # Save model B
        path_B = os.path.join(save_dir, f"{country}_Product2Patent.pt")
        torch.save(models_B[country].state_dict(), path_B)

        # Save matrix B
        matrix_B_path = path_B.replace('.pt', '_transform.npy')
        np.save(matrix_B_path, transformation_matrices_B[country])

        logger.info(f"[SAVE] {country}: Model A, B and matrices saved.")

    # TEST THE RESULTS 
    query_opportunity_product_matrix_only(
        # product_query_text="methods for directing waste gases",
        product_query_text="automotive car",
        ft_model=ft_model,
        mat_B=transformation_matrices_B['US'],
        mat_A=transformation_matrices_A['US'],
        patent_rep_dict=patent_rep_dict,
        product_rep_dict=product_rep_dict,
        firm_id_name_map=firm_id_name_map,
        firm_patent_ids=firm_patent_ids,
        patent_text_map=patent_text_map,
        top_k=5
    )

    query_opportunity_product_best(
        # product_query_text="methods for directing waste gases",
        product_query_text="computer vision and machine learning",
        ft_model=ft_model,
        model_B=models_B['US'],                 # nonlinear model from product → patent space
        model_A=models_A['US'],                 # nonlinear model from patent → product space
        patent_rep_dict=patent_rep_dict,        # {firm_id: tech field vector}
        product_rep_dict=product_rep_dict,      # {firm_id: product vector}
        firm_id_name_map=firm_id_name_map,      # {firm_id: readable company name}
        firm_patent_ids=firm_patent_ids,        # {firm_id: list of patent appln_ids}
        patent_text_map=patent_text_map,        # {appln_id: abstract preview} ==> if just want to keywords of the patent abstract, use `clean_abstract` column
        top_k=5                                 # number of top firms or fields to show
    )

