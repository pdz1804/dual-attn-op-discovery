from models.patent2product import Patent2Product
from models.product2patent import Product2Patent
from models.mlp2product import MLP2Product
from models.mlp2product import MLP2Product as MLP2Patent

MODEL_REGISTRY = {
    'linear': {
        'Patent2Product': Patent2Product,
        'Product2Patent': Product2Patent,
    },
    'mlp': {
        'Patent2Product': MLP2Product,
        'Product2Patent': MLP2Patent,
    }
    # Later you can add:
    # 'mlp': { ... }, 'transformer': { ... }, 'bilinear': { ... }
}

