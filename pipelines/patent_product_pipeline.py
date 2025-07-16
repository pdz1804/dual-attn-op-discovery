import logging
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from configs.paths import *
from configs.hyperparams import *
from models.patent2product import Patent2Product
from models.product2patent import Product2Patent
from utils.vector_utils import load_gensim_vec, text_to_vector, download_aligned_vec
from utils.model_utils import train_loop, extract_matrix, save_model_and_matrix
from inference.query_opportunity_best import query_opportunity_product_best
from inference.query_opportunity_matrix import query_opportunity_product_matrix_only

logger = logging.getLogger(__name__)

def train_pipeline():
    logger.info("[Pipeline] Start Training Patent ↔ Product")

    for country in COUNTRY:
        logger.info(f"[{country}] Training...")

        # Load fasttext vectors
        ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else 'fasttext_web_patent.vec'
        ft_model = load_gensim_vec(ft_model_path)

        # Load product embeddings
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        product_df['Product vector'] = product_df['company_keywords'].apply(lambda x: text_to_vector(x, ft_model))

        # Load patent data
        patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else './CN_JP_US_data/cleaned_patents.csv'
        patent_df = pd.read_csv(patent_file)

        patent_rep, product_rep = {}, {}
        for idx, row in product_df.iterrows():
            product_rep[row['Firm ID']] = row['Product vector']

        for firm_id, group in patent_df.groupby('hojin_id'):
            firm_id = str(firm_id)
            abstracts = group['clean_abstract'].dropna().tolist()
            tokens = []
            for abstract in abstracts:
                tokens += eval(abstract) if abstract.startswith('[') else abstract.split()
            patent_rep[firm_id] = text_to_vector('|'.join(tokens), ft_model)

        shared_ids = list(set(product_rep.keys()) & set(patent_rep.keys()))
        if len(shared_ids) == 0:
            logger.info(f"[{country}] No shared IDs. Skipping.")
            continue

        # Prepare data
        X_patent = np.stack([patent_rep[i] for i in shared_ids])
        Y_product = np.stack([product_rep[i] for i in shared_ids])

        # Train Patent → Product (Model A)
        model_A = Patent2Product()
        X_train, X_val, Y_train, Y_val = train_test_split(X_patent, Y_product, test_size=0.2, random_state=42)
        model_A, hist_A = train_loop(model_A, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32),
                                     torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32),
                                     EPOCHS_TRANS_MATRIX)
        A_matrix = extract_matrix(model_A)
        save_model_and_matrix(model_A, A_matrix, os.path.join(MODEL_DIR, f"{country}_Patent2Product.pt"))

        # Train Product → Patent (Model B)
        model_B = Product2Patent()
        X_train, X_val, Y_train, Y_val = train_test_split(Y_product, X_patent, test_size=0.2, random_state=42)
        model_B, hist_B = train_loop(model_B, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32),
                                     torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32),
                                     EPOCHS_TRANS_MATRIX)
        B_matrix = extract_matrix(model_B)
        save_model_and_matrix(model_B, B_matrix, os.path.join(MODEL_DIR, f"{country}_Product2Patent.pt"))

        # Export product embeddings
        product_df['Product embedding vector'] = product_df['Firm ID'].apply(
            lambda fid: '|'.join(map(str, model_B(torch.tensor(patent_rep[fid], dtype=torch.float32)).detach().numpy())) if fid in patent_rep else ""
        )

        product_df.to_csv(FIRM_PRODUCT_KEYWORDS, index=False)
        logger.info(f"[{country}] Exported product embeddings to {FIRM_PRODUCT_KEYWORDS}")

def test_pipeline():
    logger.info("[Pipeline] Start Testing Patent ↔ Product")

    for country in COUNTRY:
        logger.info(f"[{country}] Testing...")

        # Load fasttext vectors
        ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else 'fasttext_web_patent.vec'
        ft_model = load_gensim_vec(ft_model_path)

        # Load product data
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        product_rep = {row['Firm ID']: text_to_vector(row['company_keywords'], ft_model) for idx, row in product_df.iterrows()}

        # Load patent data
        patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else './CN_JP_US_data/cleaned_patents.csv'
        patent_df = pd.read_csv(patent_file)

        patent_rep = {}
        firm_patent_ids, patent_text_map = {}, {}
        for firm_id, group in patent_df.groupby('hojin_id'):
            firm_id = str(firm_id)
            abstracts = group['clean_abstract'].dropna().tolist()
            tokens = []
            for abstract in abstracts:
                tokens += eval(abstract) if abstract.startswith('[') else abstract.split()
            patent_rep[firm_id] = text_to_vector('|'.join(tokens), ft_model)
            firm_patent_ids[firm_id] = group['appln_id'].tolist()
            for app_id, abs_text in zip(group['appln_id'], abstracts):
                patent_text_map[app_id] = abs_text

        # Load models
        A_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Patent2Product_transform.npy"))
        B_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Product2Patent_transform.npy"))

        # Run Query Test
        query_opportunity_product_matrix_only(
            product_query_text="automotive car",
            ft_model=ft_model,
            mat_B=B_matrix,
            mat_A=A_matrix,
            patent_rep_dict=patent_rep,
            product_rep_dict=product_rep,
            firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
            firm_patent_ids=firm_patent_ids,
            patent_text_map=patent_text_map,
            top_k=5
        )

        query_opportunity_product_best(
            product_query_text="computer vision and machine learning",
            ft_model=ft_model,
            model_B=None,  # Optional: If you prefer to load and run nonlinear models here, load torch models too
            model_A=None,
            patent_rep_dict=patent_rep,
            product_rep_dict=product_rep,
            firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
            firm_patent_ids=firm_patent_ids,
            patent_text_map=patent_text_map,
            top_k=5
        )

def test_pipeline_chat():
    logger.info("[Pipeline] Start Interactive Testing (Chat Mode)")

    for country in COUNTRY:
        logger.info(f"[{country}] Setting up interactive test for {country}...")

        # Load FastText
        ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else 'fasttext_web_patent.vec'
        ft_model = load_gensim_vec(ft_model_path)

        # Load product data
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        product_rep = {row['Firm ID']: text_to_vector(row['company_keywords'], ft_model) for idx, row in product_df.iterrows()}

        # Load patent data
        patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else './CN_JP_US_data/cleaned_patents.csv'
        patent_df = pd.read_csv(patent_file)

        patent_rep = {}
        firm_patent_ids, patent_text_map = {}, {}
        for firm_id, group in patent_df.groupby('hojin_id'):
            firm_id = str(firm_id)
            abstracts = group['clean_abstract'].dropna().tolist()
            tokens = []
            for abstract in abstracts:
                tokens += eval(abstract) if abstract.startswith('[') else abstract.split()
            patent_rep[firm_id] = text_to_vector('|'.join(tokens), ft_model)
            firm_patent_ids[firm_id] = group['appln_id'].tolist()
            for app_id, abs_text in zip(group['appln_id'], abstracts):
                patent_text_map[app_id] = abs_text

        # Load matrices
        A_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Patent2Product_transform.npy"))
        B_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Product2Patent_transform.npy"))

        # Optional: load models if needed for nonlinear retrieval
        model_A_path = os.path.join(MODEL_DIR, f"{country}_Patent2Product.pt")
        model_B_path = os.path.join(MODEL_DIR, f"{country}_Product2Patent.pt")

        model_A = Patent2Product()
        model_B = Product2Patent()

        model_A.load_state_dict(torch.load(model_A_path))
        model_B.load_state_dict(torch.load(model_B_path))

        model_A.eval()
        model_B.eval()

        # Start Chat Loop
        print("\n=== Patent ↔ Product Search Chat ===")
        print(f"Country: {country}")
        print("Type 'exit' to quit.")
        print("Choose mode: 'matrix' (linear) or 'model' (nonlinear)\n")

        while True:
            user_input = input("Enter product query: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break

            mode_input = input("Choose mode (matrix/model): ").strip().lower()
            if mode_input not in ["matrix", "model"]:
                print("Invalid mode. Try again.")
                continue

            top_k = 5

            if mode_input == "matrix":
                # Use matrix-based retrieval
                query_opportunity_product_matrix_only(
                    product_query_text=user_input,
                    ft_model=ft_model,
                    mat_B=B_matrix,
                    mat_A=A_matrix,
                    patent_rep_dict=patent_rep,
                    product_rep_dict=product_rep,
                    firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
                    firm_patent_ids=firm_patent_ids,
                    patent_text_map=patent_text_map,
                    top_k=top_k
                )

            elif mode_input == "model":
                # Use model-based retrieval
                query_opportunity_product_best(
                    product_query_text=user_input,
                    ft_model=ft_model,
                    model_B=model_B,
                    model_A=model_A,
                    patent_rep_dict=patent_rep,
                    product_rep_dict=product_rep,
                    firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
                    firm_patent_ids=firm_patent_ids,
                    patent_text_map=patent_text_map,
                    top_k=top_k
                )




