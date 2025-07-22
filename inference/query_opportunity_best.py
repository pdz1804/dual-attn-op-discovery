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
    print("\n[Step 1] Embed user product query...")
    
    # Check if ft_model is an enhanced embedder or traditional FastText
    if hasattr(ft_model, 'encode_text'):
        # Enhanced embedder (sentence transformer)
        print("Using sentence transformer embedder...")
        product_vec = ft_model.encode_text(product_query_text)
    else:
        # Traditional FastText embedder
        print("Using FastText embedder...")
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
            'predicted_patent_vec': predicted_patent_vec,
            'patent_ids': patent_ids,
            'num_patents': len(patent_ids)
        })
    
    return results



