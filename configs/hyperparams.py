MAX_PAGE = 32
MAX_LEN = 864
BATCH_SIZE = 10

EPOCHS_DUAL_ATT = 10

# Config for Dual Attention Model
# Note that this is the best config so far
HIDDEN_DIM = 300
EMBEDDING_DIM = 300
LABEL_DIM = 1
ATTN_TYPE = 'dot'  
ATTN_WORD = False
ATTN_PAGE = False
SCALE = 10
PAGE_SCALE = 10

# Config for Transformer Matrix Model
COUNTRY = ['US']
EPOCHS_TRANS_MATRIX = 100

# Get only partial of the data for training only
TEST_SIZE = None
