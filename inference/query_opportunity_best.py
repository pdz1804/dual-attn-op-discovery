import torch 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def query_opportunity_product_best(
    product_query_text,
    ft_model,
    model_B, model_A,
    patent_rep_dict,
    product_rep_dict,
    firm_id_name_map,
    firm_patent_ids,
    patent_text_map,
    top_k=5,
    show_patents_per_firm=3
):
    print("\n[Step 1] Embed user product query from individual words...")
    query_tokens = product_query_text.strip().split()
    token_vecs = [ft_model[w] for w in query_tokens if w in ft_model]

    if not token_vecs:
        print("❌ None of the tokens are in FastText vocabulary.")
        return

    product_vec = np.mean(token_vecs, axis=0)
    product_tensor = torch.tensor(product_vec, dtype=torch.float32).unsqueeze(0)

    print("\n[Step 2] Predict technical field vector using model B (nonlinear)...")
    with torch.no_grad():
        patent_tensor = model_B(product_tensor)  # shape: (1, D_patent)
        predicted_patent_vec = patent_tensor.squeeze().numpy()

    print("\n[Step 3] Find top-k similar technical fields (patents)...")
    all_patent_vecs = np.stack(list(patent_rep_dict.values()))
    all_firm_ids = list(patent_rep_dict.keys())

    sims = cosine_similarity([predicted_patent_vec], all_patent_vecs)[0]
    top_k_field_idx = sims.argsort()[-top_k:][::-1]

    print("\n🔎 Top-k similar field-level patents (firm-level):")
    for rank, idx in enumerate(top_k_field_idx):
        firm_id = all_firm_ids[idx]
        firm_name = firm_id_name_map.get(firm_id, "Unknown")
        print(f"{rank+1}. Firm ID: {firm_id}, Name: {firm_name}, Cosine Sim: {sims[idx]:.4f}")
        
        patent_ids = firm_patent_ids.get(firm_id, [])
        if not patent_ids:
            print("     No associated patents found.")
            continue

        print(f"     Showing {min(show_patents_per_firm, len(patent_ids))} patent(s):")
        for p_id in patent_ids[:show_patents_per_firm]:
            abstract_preview = patent_text_map.get(p_id, "(no abstract)")
            print(f"       • appln_id: {p_id}, preview: {abstract_preview}")

    print("\n[Step 4] Map predicted patent vector back to market vector using model A (nonlinear)...")
    with torch.no_grad():
        market_tensor = model_A(patent_tensor)  # shape: (1, D_product)
        predicted_market_field_vec = market_tensor.squeeze().numpy()

    print("\n[Step 5] Find top-k companies (firm-level product vectors) similar to this predicted market field...")
    all_product_vecs = np.stack(list(product_rep_dict.values()))
    all_product_firms = list(product_rep_dict.keys())

    sims_firm = cosine_similarity([predicted_market_field_vec], all_product_vecs)[0]
    top_k_firm_idx = sims_firm.argsort()[-top_k:][::-1]

    print("\n🏢 Top-k firms likely aligned with your product:")
    for rank, idx in enumerate(top_k_firm_idx):
        firm_id = all_product_firms[idx]
        firm_name = firm_id_name_map.get(firm_id, "Unknown")
        print(f"{rank+1}. Firm ID: {firm_id}, Name: {firm_name}, Cosine Sim: {sims_firm[idx]:.4f}")



