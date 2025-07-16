import numpy as np

def select_keywords(attn_w, words, n=10):
    """Select the top-N most important words based on attention weights."""
    # Filter out zero-attention words.
    combo = [(i, j) for i, j in zip(attn_w, words) if i != 0]
    
    # Transform inputs:
    attn_w = np.array([i[0] for i in combo])
    words = [i[1] for i in combo]
    
    # Compute "attention distance" from the most attended word:
    attn_diff = attn_w.max() - attn_w
    
    # Select words whose attention is in the top-n percentile:
    attn_thres = np.percentile(attn_diff, n)
    selected_keywords = [i for i, j in zip(words, attn_diff) if j <= attn_thres]
    selected_keywords_show = [0.6 if j <= attn_thres else 0 for i, j in zip(words, attn_diff)]
    return selected_keywords, selected_keywords_show



