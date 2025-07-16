from __future__ import division

import logging 
logger = logging.getLogger(__name__)

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, data_loader):
    """Evaluate model and return metrics"""
    logger.info("[EVALUATE] Starting evaluation on dataset...")
    
    model.eval()
    gold_labels = []
    pred_probs = []
    pred_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100):
            seq_ids, num_pages, seq_lengths, label_list, hojin = batch
            
            # print("Evaluation: unique labels:", torch.unique(label_list))
            
            outputs, *_ = model(seq_ids.to(device), num_pages.to(device), seq_lengths.to(device))
            # probs = outputs.squeeze().cpu().numpy()
            probs = outputs.view(-1).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            # Make sure probs is iterable
            if np.isscalar(probs):
                probs = [probs]
                preds = [preds]

            gold_labels.extend(label_list.cpu().numpy())
            pred_probs.extend(probs)
            pred_labels.extend(preds)

    # Compute metrics
    acc = accuracy_score(gold_labels, pred_labels)
    prec = precision_score(gold_labels, pred_labels, zero_division=0)
    rec = recall_score(gold_labels, pred_labels, zero_division=0)
    f1 = f1_score(gold_labels, pred_labels, zero_division=0)
    roc = roc_auc_score(gold_labels, pred_probs)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc
    }
    
    logger.info(f"[EVALUATE] Finished evaluation.")
    logger.info(f"[EVALUATE] Metrics: {metrics}")

    return metrics


