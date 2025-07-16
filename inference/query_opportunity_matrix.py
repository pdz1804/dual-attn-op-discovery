from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def query_opportunity_product_matrix_only(
    product_query_text,
    ft_model,
    mat_B, mat_A,
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

    market_vec = np.mean(token_vecs, axis=0)  # m_est
    print(f"📐 market_vec shape: {market_vec.shape}")  # (D,)

    print("\n[Step 2] Predict technical field vector using matrix B...")
    print(f"📐 mat_B shape: {mat_B.shape}")  # (D_patent, D_product)
    predicted_tech_field_vec = market_vec @ mat_B.T  # (D_patent,)
    print(f"📐 predicted_tech_field_vec shape: {predicted_tech_field_vec.shape}")

    print("\n[Step 3] Find top-k similar technical fields (field-level vectors)...")
    all_patent_vecs = np.stack(list(patent_rep_dict.values()))  # (N_patents, D_patent)
    
    all_firm_ids = list(patent_rep_dict.keys())
    all_patent_ids = list(patent_text_map.keys())  # NEW: Get all patent IDs
    
    print(f"📐 all_patent_vecs shape: {all_patent_vecs.shape}")

    sims = cosine_similarity([predicted_tech_field_vec], all_patent_vecs)[0]  # (N_patents,)
    print(f"📐 cosine similarities to patent field vectors shape: {sims.shape}")

    top_k_field_idx = sims.argsort()[-top_k:][::-1]

    # --- old code ---
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
    
    # --- new code --- 
    # print("\n🔎 Top-k similar patents (directly at patent-level):")
    # for rank, idx in enumerate(top_k_field_idx):
    #     patent_id = all_firm_ids[idx]  # Note: all_firm_ids actually contains patent-level keys here.
    #     firm_id = patent_id.split("_")[0] if "_" in patent_id else patent_id  # Adjust if needed for mapping.
    #     firm_name = firm_id_name_map.get(firm_id, "Unknown")
    #     abstract_preview = patent_text_map.get(patent_id, "(no abstract)")

    #     print(f"{rank+1}. Patent ID: {patent_id}, Firm ID: {firm_id}, Name: {firm_name}, Cosine Sim: {sims[idx]:.4f}")
    #     print(f"       • Abstract preview: {abstract_preview}")

    print("\n[Step 4] Map predicted technical field vector back to predicted market field vector using matrix A...")
    print(f"📐 mat_A shape: {mat_A.shape}")  # (D_product, D_patent)
    predicted_market_field_vec = predicted_tech_field_vec @ mat_A.T  # (D_product,)
    print(f"📐 predicted_market_field_vec shape: {predicted_market_field_vec.shape}")

    print("\n[Step 5] Find top-k similar company market vectors (firm-level)...")
    all_product_vecs = np.stack(list(product_rep_dict.values()))  # (N_firms, D_product)
    all_firm_ids_product = list(product_rep_dict.keys())
    print(f"📐 all_product_vecs shape: {all_product_vecs.shape}")

    sims_firm = cosine_similarity([predicted_market_field_vec], all_product_vecs)[0]  # (N_firms,)
    print(f"📐 cosine similarities to firm product vectors shape: {sims_firm.shape}")

    top_k_firm_idx = sims_firm.argsort()[-top_k:][::-1]

    print("\n🏢 Top-k firms likely aligned with your product:")
    for rank, idx in enumerate(top_k_firm_idx):
        firm_id = all_firm_ids_product[idx]
        firm_name = firm_id_name_map.get(firm_id, "Unknown")
        print(f"{rank+1}. Firm ID: {firm_id}, Name: {firm_name}, Cosine Sim: {sims_firm[idx]:.4f}")
