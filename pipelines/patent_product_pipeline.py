import logging
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from ast import literal_eval
from sklearn.model_selection import train_test_split

from configs.paths import *
from configs.hyperparams import *
from models.patent2product import Patent2Product
from models.product2patent import Product2Patent
from utils.vector_utils import load_gensim_vec, text_to_vector, download_aligned_vec
from utils.model_utils import train_loop, extract_matrix, save_model_and_matrix
from utils.plot_utils import plot_train_history_trans_matrix
from inference.query_opportunity_best import query_opportunity_product_best
from inference.query_opportunity_matrix import query_opportunity_product_matrix_only

logger = logging.getLogger(__name__)

def save_representations_to_json(rep_dict, file_path):
    """Save a dictionary of representations to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_dict = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in rep_dict.items()}
    with open(file_path, 'w') as f:
        json.dump(serializable_dict, f)
    logger.info(f"Saved representations to {file_path}")

def load_representations_from_json(file_path):
    """Load a dictionary of representations from a JSON file."""
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} does not exist. Returning empty dict.")
        return {}
    with open(file_path, 'r') as f:
        serializable_dict = json.load(f)
    # Convert lists back to numpy arrays
    rep_dict = {key: np.array(value) if isinstance(value, list) else value for key, value in serializable_dict.items()}
    logger.info(f"Loaded representations from {file_path} with {len(rep_dict)} entries")
    return rep_dict

def process_representations(country, product_df, patent_df, ft_model, data_dir):
    """Process and save patent and product representations."""
    patent_rep, product_rep = {}, {}
    
    # Process product representations
    for idx, row in tqdm(product_df.iterrows(), total=len(product_df), desc=f"[{country}] Processing products"):
        product_rep[row['Firm ID']] = row['Product vector']
    logger.info(f"[{country}] Prepared product representations for {len(product_rep)} firms")
    
    # Save product representations to JSON
    product_json_path = os.path.join(data_dir, f"{country}_product_rep.json")
    save_representations_to_json(product_rep, product_json_path)

    # Process patent representations
    firm_count = 0
    for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patents"):
        firm_id = str(firm_id)
        abstracts = group['clean_abstract'].dropna().tolist()
        tokens = []
        for abstract in abstracts:
            tokens += eval(abstract) if abstract.startswith('[') else abstract.split()
        patent_rep[firm_id] = text_to_vector('|'.join(tokens), ft_model)
        
        # Log details for the first two firms
        if firm_count < 2:
            logger.info(f"[{country}] Processed firm {firm_id}: Tokens={tokens[:50]}... | Abstracts={len(abstracts)}")
            firm_count += 1
    
    logger.info(f"[{country}] Prepared patent representations for {len(patent_rep)} firms")
    
    # Save patent representations to JSON
    patent_json_path = os.path.join(data_dir, f"{country}_patent_rep.json")
    save_representations_to_json(patent_rep, patent_json_path)
    
    return patent_rep, product_rep

def train_pipeline():
    logger.info("[Pipeline] Start Training Patent ↔ Product")

    for country in COUNTRY:
        logger.info(f"[{country}] Training...")

        # Load fasttext vectors
        ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else FASTTEXT_VEC
        ft_model = load_gensim_vec(ft_model_path)
        logger.info(f"[{country}] Loaded FastText model from {ft_model_path}")

        # Load product embeddings
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        product_df['Product vector'] = product_df['company_keywords'].apply(lambda x: text_to_vector(x, ft_model))
        logger.info(f"[{country}] Loaded product embeddings with {len(product_df)} firms")

        # Load patent data
        patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else US_PATENT_DATA_CLEANED
        
        if not TEST_SIZE:
            patent_df = pd.read_csv(patent_file)
        else:
            patent_df = pd.read_csv(patent_file).sample(frac=TEST_SIZE, random_state=42)
        
        logger.info(f"[{country}] Loaded patent data with {len(patent_df)} records")

        # Process and save representations
        patent_rep, product_rep = process_representations(country, product_df, patent_df, ft_model, DATA_DIR)
        logger.info(f"[{country}] Processed representations saved to JSON files")
        
        shared_ids = list(set(product_rep.keys()) & set(patent_rep.keys()))
        if len(shared_ids) == 0:
            logger.info(f"[{country}] No shared IDs. Skipping.")
            continue

        # Prepare data
        X_patent = np.stack([patent_rep[i] for i in shared_ids])
        Y_product = np.stack([product_rep[i] for i in shared_ids])
        logger.info(f"[{country}] Prepared training data with {len(X_patent)} shared firms")
        
        if len(X_patent) == 0 or len(Y_product) == 0:
            logger.info(f"[{country}] No data to train. Skipping.")
            continue

        # Train Patent → Product (Model A)
        model_A = Patent2Product()
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_patent, Y_product, test_size=0.2, random_state=42)
        logger.info(f"[{country}] Training Patent2Product model with {len(X_train)} training samples")
        logger.info(f"[{country}] Validation size: {len(X_val)} samples")
        
        model_A, hist_A = train_loop(model_A, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32),
                                     torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32),
                                     EPOCHS_TRANS_MATRIX)
        logger.info(f"[{country}] Finished training Patent2Product model")
        
        # Plot training history for Model A
        plot_train_history_trans_matrix(hist_A, "Patent2Product", country)
        
        # Save model and extract matrix
        A_matrix = extract_matrix(model_A)
        save_model_and_matrix(model_A, A_matrix, os.path.join(MODEL_DIR, f"{country}_Patent2Product.pt"))

        # Train Product → Patent (Model B)
        model_B = Product2Patent()
        X_train, X_val, Y_train, Y_val = train_test_split(Y_product, X_patent, test_size=0.2, random_state=42)
        logger.info(f"[{country}] Training Product2Patent model with {len(X_train)} training samples")
        logger.info(f"[{country}] Validation size: {len(X_val)} samples")
        
        model_B, hist_B = train_loop(model_B, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32),
                                     torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32),
                                     EPOCHS_TRANS_MATRIX)
        logger.info(f"[{country}] Finished training Product2Patent model")
        
        # Plot training history for Model B
        plot_train_history_trans_matrix(hist_B, "Product2Patent", country)
        
        # Save model and extract matrix
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
        ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else FASTTEXT_VEC
        ft_model = load_gensim_vec(ft_model_path)

        # Load product data
        # In test_pipeline and test_pipeline_chat
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        
        data_path = US_WEB_DATA
        us_web_with_company = pd.read_csv(data_path)
        
        # Keep only necessary columns from `us_web_with_company`
        company_name_map = us_web_with_company[['hojin_id', 'company_name']].drop_duplicates()

        # Merge based on hojin_id
        product_df = product_df.merge(company_name_map, on='hojin_id', how='left')
        
        product_json_path = os.path.join(DATA_DIR, f"{country}_product_rep.json")
        product_rep = load_representations_from_json(product_json_path)
        if not product_rep:
            logger.warning(f"[{country}] No product representations loaded. Computing from scratch.")
            product_rep = {row['Firm ID']: text_to_vector(row['company_keywords'], ft_model) 
                        for idx, row in tqdm(product_df.iterrows(), total=len(product_df), desc=f"[{country}] Processing products")}
            
        # Load patent data
        patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else US_PATENT_DATA_CLEANED
        patent_df = pd.read_csv(patent_file)

        # Load patent representations from JSON
        patent_json_path = os.path.join(DATA_DIR, f"{country}_patent_rep.json")
        patent_rep = load_representations_from_json(patent_json_path)
        firm_patent_ids, patent_text_map = {}, {}
        
        # Always compute firm_patent_ids and patent_text_map
        for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patent IDs"):
            firm_id = str(firm_id)
            firm_patent_ids[firm_id] = group['appln_id'].tolist()
            for app_id, abs_text in zip(group['appln_id'], group['clean_abstract'].dropna()):
                patent_text_map[app_id] = abs_text

        # Compute patent representations if not loaded
        if not patent_rep:
            logger.warning(f"[{country}] No patent representations loaded. Computing from scratch.")
            patent_rep = {}
            for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patents"):
                firm_id = str(firm_id)
                abstracts = group['clean_abstract'].dropna().tolist()
                tokens = []
                for abstract in abstracts:
                    tokens += literal_eval(abstract) if abstract.startswith('[') else abstract.split()
                patent_rep[firm_id] = text_to_vector('|'.join(tokens), ft_model)

        logger.info(f"[{country}] Prepared {len(firm_patent_ids)} firm patent IDs and {len(patent_text_map)} patent text mappings")
        
        # Load models
        A_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Patent2Product_transform.npy"))
        B_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Product2Patent_transform.npy"))

        # Load models for nonlinear query
        model_A_path = os.path.join(MODEL_DIR, f"{country}_Patent2Product.pt")
        model_B_path = os.path.join(MODEL_DIR, f"{country}_Product2Patent.pt")
        model_A = Patent2Product()
        model_B = Product2Patent()
        model_A.load_state_dict(torch.load(model_A_path))
        model_B.load_state_dict(torch.load(model_B_path))
        model_A.eval()
        model_B.eval()

        # Run Query Test
        logger.info(f"[{country}] Running matrix query for 'chemical waste machine'")
        results_matrix = query_opportunity_product_matrix_only(
            product_query_text="chemical waste machine",
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
        logger.info(f"[{country}] Matrix query returned results")

        logger.info(f"[{country}] Running model query for 'computer vision and machine learning'")
        results_best = query_opportunity_product_best(
            product_query_text="computer vision and machine learning",
            ft_model=ft_model,
            model_B=model_B,
            model_A=model_A,
            patent_rep_dict=patent_rep,
            product_rep_dict=product_rep,
            firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
            firm_patent_ids=firm_patent_ids,
            patent_text_map=patent_text_map,
            top_k=5
        )
        logger.info(f"[{country}] Model query returned results")

def test_pipeline_chat():
    logger.info("[Pipeline] Start Interactive Testing (Chat Mode)")

    for country in COUNTRY:
        logger.info(f"[{country}] Setting up interactive test for {country}...")

        # Load FastText
        ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else FASTTEXT_VEC
        ft_model = load_gensim_vec(ft_model_path)

        # In test_pipeline and test_pipeline_chat
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        
        data_path = US_WEB_DATA
        us_web_with_company = pd.read_csv(data_path)
        
        # Keep only necessary columns from `us_web_with_company`
        company_name_map = us_web_with_company[['hojin_id', 'company_name']].drop_duplicates()

        # Merge based on hojin_id
        product_df = product_df.merge(company_name_map, on='hojin_id', how='left')
        
        product_json_path = os.path.join(DATA_DIR, f"{country}_product_rep.json")
        product_rep = load_representations_from_json(product_json_path)
        if not product_rep:
            logger.warning(f"[{country}] No product representations loaded. Computing from scratch.")
            product_rep = {row['Firm ID']: text_to_vector(row['company_keywords'], ft_model) 
                        for idx, row in tqdm(product_df.iterrows(), total=len(product_df), desc=f"[{country}] Processing products")}

        # Load patent data
        patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else US_PATENT_DATA_CLEANED
        patent_df = pd.read_csv(patent_file)

        # Load patent representations from JSON
        patent_json_path = os.path.join(DATA_DIR, f"{country}_patent_rep.json")
        patent_rep = load_representations_from_json(patent_json_path)
        firm_patent_ids, patent_text_map = {}, {}
        
        # Always compute firm_patent_ids and patent_text_map
        for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patent IDs"):
            firm_id = str(firm_id)
            firm_patent_ids[firm_id] = group['appln_id'].tolist()
            for app_id, abs_text in zip(group['appln_id'], group['clean_abstract'].dropna()):
                patent_text_map[app_id] = abs_text

        # Compute patent representations if not loaded
        if not patent_rep:
            logger.warning(f"[{country}] No patent representations loaded. Computing from scratch.")
            patent_rep = {}
            for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patents"):
                firm_id = str(firm_id)
                abstracts = group['clean_abstract'].dropna().tolist()
                tokens = []
                for abstract in abstracts:
                    tokens += literal_eval(abstract) if abstract.startswith('[') else abstract.split()
                patent_rep[firm_id] = text_to_vector('|'.join(tokens), ft_model)

        logger.info(f"[{country}] Prepared {len(firm_patent_ids)} firm patent IDs and {len(patent_text_map)} patent text mappings")

        # Load matrices
        A_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Patent2Product_transform.npy"))
        B_matrix = np.load(os.path.join(MODEL_DIR, f"{country}_Product2Patent_transform.npy"))

        # Load models for nonlinear retrieval
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
            logger.info(f"[{country}] Processing query: '{user_input}' in {mode_input} mode")
            
            if mode_input == "matrix":
                logger.info(f"[{country}] Running matrix query for '{user_input}'")
                results_matrix = query_opportunity_product_matrix_only(
                    product_query_text=user_input.lower(),
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
                logger.info(f"[{country}] Matrix query returned results")

            elif mode_input == "model":
                logger.info(f"[{country}] Running model query for '{user_input}'")
                results_best = query_opportunity_product_best(
                    product_query_text=user_input.lower(),
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
                logger.info(f"[{country}] Model query returned results")




