import torch
import torch.nn as nn
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def train_loop(model, X_train, Y_train, X_val, Y_val, epochs, lr=1e-3):
    logger.info("Starting training loop of transformer matrix model")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "train_cos": [], "val_cos": []}

    logger.info(f"Training with {epochs} epochs, learning rate={lr}")
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

# # old 
# def extract_matrix(model):
#     W1 = model.net[0].weight.detach().cpu().numpy()
#     W2 = model.net[2].weight.detach().cpu().numpy()
#     matrix = W2 @ W1
#     logger.info(f"Extracted matrix with shape {matrix.shape}")
#     return matrix

# new 
def extract_matrix_by_sampling(model, input_dim, num_samples=10000):
    logger.info("Estimating transformation matrix via sampling")

    X = torch.randn(num_samples, input_dim)
    with torch.no_grad():
        Y = model(X)

    # Solve least squares: find M such that Y ≈ X @ M.T
    # So M ≈ (X^T X)^-1 X^T Y
    X_np = X.numpy()
    Y_np = Y.numpy()
    pseudo_inverse = np.linalg.pinv(X_np)
    M = pseudo_inverse @ Y_np  # shape: (input_dim, output_dim)
    logger.info(f"Estimated matrix shape: {M.shape}")
    return M.T  # to match shape of dim_out x dim_in

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

# new approach using polynomial approximation
def extract_polynomial_relationship(model, input_dim, degree=2, num_samples=10000):
    X = torch.randn(num_samples, input_dim)
    with torch.no_grad():
        Y = model(X)

    X_np = X.numpy()
    Y_np = Y.numpy()

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_np)

    reg = Ridge(alpha=1e-3)
    reg.fit(X_poly, Y_np)

    logger.info(f"Fitted polynomial of degree {degree} with shape {reg.coef_.shape}")
    return reg, poly

# new approach using kernel approximation
def extract_kernel_relationship(model, input_dim, num_samples=2000, kernel='rbf'):
    X = torch.randn(num_samples, input_dim)
    with torch.no_grad():
        Y = model(X)

    X_np = X.numpy()
    Y_np = Y.numpy()

    reg = KernelRidge(kernel=kernel, alpha=1e-3)
    reg.fit(X_np, Y_np)

    logger.info(f"Fitted Kernel Ridge model with kernel={kernel}")
    return reg

def extract_matrix(model, model_type='linear', dim=300, approx_method='sampling'):
    """
    Extract a transformation matrix from a model.
    - For linear: exact.
    - For MLP: sampled approximation.
    - For other nonlinear: raise warning or skip.
    """
    if model_type == 'linear':
        layers = [m for m in model.net if isinstance(m, nn.Linear)]
        if len(layers) >= 2:
            W1 = layers[0].weight.detach().cpu().numpy()
            W2 = layers[1].weight.detach().cpu().numpy()
            matrix = W2 @ W1
            logger.info(f"[{model_type}] Extracted matrix shape: {matrix.shape}")
            return matrix
        else:
            logger.warning(f"[{model_type}] Not enough linear layers to extract matrix.")
            return None

    elif model_type == 'mlp':
        if approx_method == 'sampling':
            logger.info(f"[{model_type}] Using sampled approximation to extract matrix.")
            return extract_matrix_by_sampling(model, input_dim=dim)
        elif approx_method == 'polynomial':
            logger.info(f"[{model_type}] Using polynomial approximation to extract matrix.")
            return extract_polynomial_relationship(model, input_dim=dim)
        elif approx_method == 'kernel':
            logger.info(f"[{model_type}] Using kernel approximation to extract matrix.")
            return extract_kernel_relationship(model, input_dim=dim)
        else:
            logger.warning(f"[{model_type}] Unknown approximation method: {approx_method}")
            return None

    else:
        logger.warning(f"[{model_type}] Matrix extraction is not supported for this architecture.")
        return None

def save_model_and_matrix(model, matrix, model_path):
    torch.save(model.state_dict(), model_path)
    np.save(model_path.replace('.pt', '_transform.npy'), matrix)
    logger.info(f"Saved model to {model_path} and matrix to {model_path.replace('.pt', '_transform.npy')}")



