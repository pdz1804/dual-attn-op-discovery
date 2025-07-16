import torch
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def train_loop(model, X_train, Y_train, X_val, Y_val, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "train_cos": [], "val_cos": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val)

            cos_train = torch.nn.functional.cosine_similarity(pred, Y_train, dim=1).mean().item()
            cos_val = torch.nn.functional.cosine_similarity(val_pred, Y_val, dim=1).mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_cos"].append(cos_train)
        history["val_cos"].append(cos_val)

        logger.info(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, CosTrain={cos_train:.4f}, CosVal={cos_val:.4f}")

    return model, history

def extract_matrix(model):
    W1 = model.net[0].weight.detach().cpu().numpy()
    W2 = model.net[2].weight.detach().cpu().numpy()
    return W2 @ W1

def save_model_and_matrix(model, matrix, model_path):
    torch.save(model.state_dict(), model_path)
    np.save(model_path.replace('.pt', '_transform.npy'), matrix)
    logger.info(f"Saved model to {model_path} and matrix to {model_path.replace('.pt', '_transform.npy')}")



