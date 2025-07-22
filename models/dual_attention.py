from __future__ import division
import logging 
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from models.layers.layer_norm import LayerNorm
from models.layers.sparsemax import Sparsemax

class DualAttnModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, label_dim=1, scale=10, page_scale=10, attn_type='dot', attn_word=False, attn_page=False):
        super(DualAttnModel, self).__init__()
        
        self.attn_word = attn_word  # <<< NEW TOGGLE
        self.attn_page = attn_page  # <<< NEW TOGGLE
        
        # Add LayerNorm for normalization
        self.ln_word = LayerNorm(hidden_dim)
        self.ln_page = LayerNorm(hidden_dim)
        
        # Embedding Layer: Word embedding lookup for input tokens
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        
        # Feedforward + Dropout + Attention Parameters
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.affine = nn.Linear(embed_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()                                     # for final output (binary classifier).
        self.softmax = nn.Softmax(dim=-1)                               # Softmax for word-level attention
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention Parameters (Word and Page)
        self.scale = scale
        self.page_scale = page_scale
        
        if not attn_word:
            self.V = nn.Parameter(torch.randn(hidden_dim, 1))             # word-level attention
        else: 
            # Optional: this replaces the use of self.V with a multi-head attention layer
            # self.multi_head = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
            
            # Optional 2: Manual linear projections (optional, or just use raw embeddings)
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if not attn_page:
            self.W = nn.Parameter(torch.randn(hidden_dim, 1))               # page-level attention
        else:
            self.page_q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.page_k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.page_v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Classifier and Attention Control
        self.decoder = nn.Linear(hidden_dim, label_dim, bias=False)
        self.attn_type = attn_type
        self.page_sparsemax = Sparsemax(dim=-1)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype, device=q.device))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))  # mask shape: (B, L)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn

    def compute_attention(self, inputs, vector, scale, mask, attn_mode='softmax', return_scores=False):
        # inputs: (B, ..., D), vector: (D, 1), mask: same shape as attention score
        raw_scores = torch.matmul(inputs, vector).squeeze(-1) / scale
        masked_scores = raw_scores.masked_fill(mask.bool(), -9999)

        if attn_mode == 'softmax':
            attn = self.softmax(masked_scores)
        elif attn_mode == 'sparsemax':
            attn = self.page_sparsemax(masked_scores)
        else:
            raise ValueError(f"Unknown attention type: {attn_mode}")
        
        return (attn, raw_scores) if return_scores else attn

    def forward(self, seq_ids, num_pages, seq_lengths=None):
        # print("\n===== FORWARD PASS DEBUG =====")
        # logger.info(f"seq_ids shape: {seq_ids.shape}")
        # logger.info(f"seq_ids max: {seq_ids.max().item()}, min: {seq_ids.min().item()}")
        # logger.info(f"Embedding vocab_size: {self.embeddings.num_embeddings}")
        
        # Embedding lookup + shape
        seq_embs = self.embeddings(seq_ids)                             # seq_ids shape: (batch_size, max_page, max_len)
        
        # print("seq_embs:", seq_embs.shape, "mean:", seq_embs.mean().item(), "std:", seq_embs.std().item())
        
        seq_embs = self.dropout(seq_embs)                               # Apply dropout to embeddings
        batch_size , max_page, max_len, hidden_dim = seq_embs.size()    # batch_size(#comp), #webpages, #words, hidden_dim
        hidden_vecs = seq_embs
        # hidden_vecs = self.affine(seq_embs)                             # Now (B, P, L, hidden_dim=512)
            
        # -----<token-level attention>-----
        # Word-Level Attention (per page)
        inter_out = hidden_vecs if self.attn_type == 'dot' else torch.tanh(self.attn_linear(hidden_vecs))

        # Masking: vectorized
        seq_range = torch.arange(max_len).unsqueeze(0).unsqueeze(0).expand(batch_size, max_page, max_len).to(seq_ids.device)
        seq_len_exp = seq_lengths.unsqueeze(-1).expand_as(seq_range)
        word_mask = seq_range >= seq_len_exp

        if not self.attn_word: 
            word_attn = self.compute_attention(inter_out, self.V, self.scale, word_mask, attn_mode='softmax').unsqueeze(2)
            # webpage_vec = torch.sum(word_attn * hidden_vecs, dim=2)
            webpage_vec = torch.einsum('abcd, abde -> abe', word_attn, hidden_vecs)
        else:
            # Flatten for MHA: (B * P, L, D)
            flat_vecs = hidden_vecs.view(-1, max_len, hidden_dim)
            flat_mask = word_mask.view(-1, max_len)
            
            # Safeguard against fully-masked inputs
            all_masked_rows = flat_mask.sum(dim=1) == flat_mask.size(1)
            
            if all_masked_rows.any():
                # print("[WARNING] Fully masked sequences detected in MHA input — patching...")
                flat_mask[all_masked_rows] = False

            # # multi-head self-attention (Q=K=V=flat_vecs)
            # mha_out, _ = self.multi_head(flat_vecs, flat_vecs, flat_vecs, key_padding_mask=flat_mask)
            
            q = self.q_proj(flat_vecs)
            k = self.k_proj(flat_vecs)
            v = self.v_proj(flat_vecs)

            # Optional mask fix
            attn_mask = flat_mask.bool() if flat_mask is not None else None

            attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask=attn_mask)
            
            # if torch.isnan(mha_out).any():
            #     print("[DEBUG] NaNs detected in MHA output!")
            #     print("flat_vecs stats:", flat_vecs.mean().item(), flat_vecs.std().item())
            #     print("flat_mask sum:", flat_mask.sum(dim=1))  # Check how many tokens unmasked
            #     raise ValueError("NaNs in MHA output — likely due to all-masked input.")

            # Mean pooling over tokens to get each page vector
            # Old 
            webpage_vec = attn_output.mean(dim=1).view(batch_size, max_page, hidden_dim)
            
            # New
            # Reshape attn_weights back: (B*P, L, L) → we take attention of [CLS] or mean across heads
            word_attn = attn_weights.view(batch_size, max_page, max_len, max_len)  # [B, P, Lq, Lk]
            word_attn = word_attn.mean(dim=2)  # [B, P, Lk] ← average attention received by each token
            word_attn = word_attn.unsqueeze(2)  # match the shape [B, P, 1, L] if needed
        
        # print("webpage_vec shape:", webpage_vec.shape, "mean:", webpage_vec.mean().item())
        
        # Uncomment if you want to apply LayerNorm after word-level attention
        # Option 1: Apply LayerNorm after word-level attention
        # webpage_vec = self.ln_word(webpage_vec)
        
        # Option 2: Add residual connection
        if self.attn_word:
            # Only add residual if webpage_vec and residual_vec come from different paths
            residual_vec = torch.einsum('abcd, abde -> abe', word_attn, hidden_vecs)
            webpage_vec = self.ln_word(webpage_vec + residual_vec)
        else:
            # No need to add it — already same as webpage_vec
            webpage_vec = self.ln_word(webpage_vec)
        
        #-----<page-level attention>-----
        # Masking: vectorized
        page_range = torch.arange(max_page).unsqueeze(0).expand(batch_size, max_page).to(seq_ids.device)
        num_pages_exp = num_pages.unsqueeze(1).expand_as(page_range)
        page_mask = page_range >= num_pages_exp

        if not self.attn_page:
            # page_attn = self.compute_attention(webpage_vec, self.W, self.page_scale, page_mask, attn_mode='sparsemax').unsqueeze(1)
            page_attn, page_scores = self.compute_attention(webpage_vec, self.W, self.page_scale, page_mask, attn_mode='sparsemax', return_scores=True)
            page_attn = page_attn.unsqueeze(1)
            final_vec = torch.bmm(page_attn, webpage_vec).squeeze(1)
        else:
            # Project page representations
            page_q = self.page_q_proj(webpage_vec)  # (B, P, D)
            page_k = self.page_k_proj(webpage_vec)
            page_v = self.page_v_proj(webpage_vec)

            # Compute page attention using scaled dot product
            page_scores = torch.matmul(page_q, page_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(hidden_dim, dtype=page_q.dtype, device=page_q.device))

            # Apply masking
            page_scores = page_scores.masked_fill(page_mask.unsqueeze(1), float('-inf'))

            # Softmax over pages
            page_attn = torch.softmax(page_scores, dim=-1)  # (B, Pq=pages, Pk=pages)

            # Weighted combination of page values
            final_vec = torch.matmul(page_attn, page_v).mean(dim=1)  # mean across pages (or use [:,0] for [CLS]-like)

        # Uncomment if you want to apply LayerNorm after page-level attention
        # Old 
        final_vec = self.ln_page(final_vec)
        # print("final_vec shape:", final_vec.shape)
        
        # New: residual connection
        # - final_vec is a weighted sum (attention) over pages
        # - residual_page_vec is a raw sum over all pages (ignores page_attn)
        # residual_page_vec = webpage_vec.sum(dim=1)  # (B, D)
        # final_vec = self.ln_page(final_vec + residual_page_vec)
        
        # Final classification vector c passed through linear decoder and sigmoid → binary probability.
        final_vec = self.dropout(final_vec)
        senti_scores = self.decoder(final_vec)
        probs = self.sigmoid(senti_scores)
        
        assert not torch.isnan(webpage_vec).any(), "NaN in webpage_vec"
        assert not torch.isnan(final_vec).any(), "NaN in final_vec"
        assert not torch.isnan(senti_scores).any(), "NaN in senti_scores"
        assert not torch.isnan(probs).any(), "NaN in probs"
        
        # print("probs stats: min =", probs.min().item(), "max =", probs.max().item())
        
        return probs, senti_scores, word_attn, page_attn, final_vec, page_scores, webpage_vec
        # return senti_scores, senti_scores, word_attn, page_attn, final_vec, page_scores, webpage_vec
    
    def load_vector(self, pretrained_vectors, trainable=False):
        '''
        Load pre-savedd word embeddings
        '''
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_vectors))
        self.embeddings.weight.requires_grad = trainable
        print('Dual Att Model: Embeddings loaded')




