import logging
logger = logging.getLogger(__name__)

import torch
import time
from tqdm import tqdm
from configs import hyperparams
from models.dual_attention import DualAttnModel
from training.evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader, val_dataloader, optimizer, loss_function, num_epochs=10):
    """Training loop"""
    start_time = time.time()
    # Training loop...
    
    logger.info("[TRAIN] Starting training loop...")
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": []
    }

    best_model = None
    best_val_acc = 0

    for epoch in range(num_epochs):
        logger.info(f"\n[TRAIN] Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 50)
        logger.info(f"[TRAIN] Time elapsed: {time.time() - start_time:.2f} seconds")
        
        model.train()
        total_loss = 0
        count = 0

        for batch in tqdm(train_dataloader, desc="Training", ncols=100):
            seq_ids, num_pages, seq_lengths, label_list, hojin = batch

            optimizer.zero_grad()
            
            # print(seq_ids.shape)
            # print(num_pages.shape)
            # print(seq_lengths.shape)            
            
            # ---- old ----
            preds, *_ = model(seq_ids.to(device), num_pages.to(device), seq_lengths.to(device))
            
            # print(preds.shape)
            # print(label_list.shape)
            
            # print("Raw labels:", label_list)
            # print("Unique labels:", torch.unique(label_list))
            # print("Model output (probs) min/max:", preds.min().item(), preds.max().item())
            
            # Before computing loss
            assert label_list.max() <= 1 and label_list.min() >= 0, \
                f"[ERROR] label_list must be in [0,1], but got min={label_list.min()} max={label_list.max()}"
            
            # loss = loss_function(preds.squeeze(), label_list.to(device).float())
            loss = loss_function(preds.view(-1), label_list.to(device).float().view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * seq_ids.size(0)
            count += seq_ids.size(0)

        avg_train_loss  = total_loss / count
        history["train_loss"].append(avg_train_loss )
        
        logger.info(f"[TRAIN] Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Evaluate on training and validation set
        train_metrics = evaluate(model, train_dataloader)
        
        # Evaluate on validation set & compute val loss
        model.eval()
        val_loss_total = 0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Computing Val Loss", ncols=100):
                seq_ids, num_pages, seq_lengths, label_list, hojin = batch
                preds, *_ = model(seq_ids.to(device), num_pages.to(device), seq_lengths.to(device))
                
                # loss = loss_function(preds.squeeze(), label_list.to(device).float())
                loss = loss_function(preds.view(-1), label_list.to(device).float())
                
                val_loss_total += loss.item() * seq_ids.size(0)
                val_count += seq_ids.size(0)

        avg_val_loss = val_loss_total / val_count
        history["val_loss"].append(avg_val_loss)
        
        logger.info(f"[VAL] Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        val_metrics = evaluate(model, val_dataloader)

        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)
        
        logger.info(f"[METRICS] Train Metrics: {train_metrics}")
        logger.info(f"[METRICS] Val Metrics  : {val_metrics}")

        print(f"\nTrain Loss: {avg_train_loss:.4f}")
        print(f"Val Loss  : {avg_val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Val Metrics  : {val_metrics}")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            logger.info(f"[CHECKPOINT] New best model at Epoch {epoch+1} with Val Acc {val_metrics['accuracy']:.4f}")
            best_val_acc = val_metrics["accuracy"]
            best_model = model.state_dict()

    logger.info("\n[TRAIN] Training complete.")
    return best_model, history


