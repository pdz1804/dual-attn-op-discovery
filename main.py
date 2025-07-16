# Import necessary libraries
from __future__ import division
import logging
import sys
import argparse

# === Create a fresh log file every time ===
LOG_FILE_PATH = "pdzttb.log"

# Remove any existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up new logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),  # Overwrite each run
        logging.StreamHandler(sys.stdout),  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# Import from files 
from configs.paths import *
from configs.hyperparams import *
from utils.seed_everything import set_seed
from pipelines import dual_attention_pipeline, patent_product_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, choices=['dual_attn', 'patent_product'], required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'chat'], default=None, help="Train/Test/Chat mode (only for patent_product pipeline)")

    args = parser.parse_args()
    
    set_seed(42)  # Set seed for reproducibility

    if args.pipeline == 'dual_attn':
        dual_attention_pipeline.run()

    elif args.pipeline == 'patent_product':
        if args.mode is None:
            raise ValueError("Error: --mode is required for patent_product pipeline. Use --mode=train / test / chat.")

        if args.mode == 'train':
            patent_product_pipeline.train_pipeline()
        elif args.mode == 'test':
            patent_product_pipeline.test_pipeline()
        elif args.mode == 'chat':
            patent_product_pipeline.test_pipeline_chat()

if __name__ == "__main__":
    # your pipeline code here
    main()
    exit(1)  # Exit after running the pipeline
    