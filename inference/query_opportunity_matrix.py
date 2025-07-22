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
    print("\n[Step 1] Embed user product query...")
    
    # Check if ft_model is an enhanced embedder or traditional FastText
    if hasattr(ft_model, 'encode_text'):
        # Enhanced embedder (sentence transformer)
        print("Using sentence transformer embedder...")
        market_vec = ft_model.encode_text(product_query_text)
    else:
        # Traditional FastText embedder
        print("Using FastText embedder...")
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
        
        print(f"     Associated Patents ({len(patent_ids)} patents):")
        for j, patent_id in enumerate(patent_ids[:show_patents_per_firm]):
            abstract = patent_text_map.get(patent_id, "No abstract available")
            # Truncate abstract for display
            abstract_preview = abstract[:100] + "..." if len(abstract) > 100 else abstract
            print(f"       {j+1}. Patent ID: {patent_id}")
            print(f"          Abstract: {abstract_preview}")
        
        if len(patent_ids) > show_patents_per_firm:
            print(f"       ... and {len(patent_ids) - show_patents_per_firm} more patents.")
    
    print(f"\n✅ Found {len(top_k_field_idx)} top companies and their associated patents.")

    # Return results for further processing if needed
    results = []
    for rank, idx in enumerate(top_k_field_idx):
        firm_id = all_firm_ids[idx]
        firm_name = firm_id_name_map.get(firm_id, "Unknown")
        patent_ids = firm_patent_ids.get(firm_id, [])
        
        results.append({
            'rank': rank + 1,
            'firm_id': firm_id,
            'firm_name': firm_name,
            'cosine_similarity': sims[idx],
            'patent_ids': patent_ids,
            'num_patents': len(patent_ids)
        })
    
    return results
