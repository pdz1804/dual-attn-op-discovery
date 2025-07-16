# pipelines/dual_attention_pipeline.py

import logging
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.layers.fast_vector import FastVector
from models.dual_attention import DualAttnModel
from data_loader.web_data_preprocess import clean_tokens
from data_loader.tokenizer import Tokenizer
from utils.count_params import count_parameters
from utils.colorize_attention import colorize
from utils.select_keywords import select_keywords
from utils.plot_utils import plot_loss, plot_accuracy
from utils.seed_everything import set_seed
from training.train_dual_attention import train
from training.evaluate import evaluate
from configs.paths import *
from configs.hyperparams import *

def run():
    logger = logging.getLogger(__name__)
    logger.info("[DualAttention] Starting pipeline...")
    
    set_seed(42)  # Set seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Data Loading ===
    logger.info("[STEP 1] Loading data...")
    
    if not TEST_SIZE:
        data = pd.read_csv(US_WEB_DATA)
    else:
        data = pd.read_csv(US_WEB_DATA).sample(frac=TEST_SIZE, random_state=42)
        
    logger.info(f"Data shape after loading: {data.shape}")
    
    # Apply token cleaning
    logger.info("Cleaning text data...")
    data['cleaned'] = data['text'].progress_apply(clean_tokens)
    data = data[['hojin_id', 'company_name', 'url', 'cleaned', 'hightechflag']]
    data = data.rename(columns={'cleaned': 'cleaned_content', 'url': 'urls'})
    logger.info(f"Data shape after column selection: {data.shape}")

    # === Filter short data and prepare tokenizer ===
    all_words = [word for text in data['cleaned_content'] for word in text.split('|')]
    vocab = list(set(all_words))
    
    logger.info(f"Unique words in cleaned content: {len(vocab)}")

    # Load the clean fasttext model
    # vec_path = r'C:\PDZ\Intern\Shared_2025\Phase 2\reduced_fasttext.vec'
    vec_path = FASTTEXT_VEC
    en_dictionary_reduced = FastVector(vector_file=vec_path)

    words = list(en_dictionary_reduced.word2id.keys())
    vectors = np.array([en_dictionary_reduced[word] for word in words])
    wv_dict= dict(zip(words, vectors))
    
    logger.info(f"Unique words in fasttext vectors: {len(words)}")

    # Process the data here 
    no_values = []

    for i in tqdm(data.cleaned_content):
        try:
            i = i.split('|')
            i = [j for j in i if j in wv_dict]
            if len(i) < 1:
                no_values.append(1)
            else:
                no_values.append(0)
        except:
            no_values.append(1)

    data['no_values'] = no_values
    data = data[data.no_values == 0]
    data = data[['hojin_id', 'company_name', 'urls', 'cleaned_content', 'hightechflag']]

    hojin_ids = list(set(data.hojin_id))

    sample_data = pd.DataFrame({})

    for hojin_id in tqdm(hojin_ids):
        temp = data[data.hojin_id == hojin_id]
        if temp.shape[0] <= MAX_PAGE:
            sample_data = pd.concat([sample_data, temp], ignore_index=True)
        else:
            temp = temp.sample(n=MAX_PAGE)
            sample_data = pd.concat([sample_data, temp], ignore_index=True)
            # sample_data = pd.concat([sample_data, temp.iloc[:MAX_PAGE, :]], ignore_index=True)

    num_words = [len(i.split('|')) for i in sample_data.cleaned_content]
    sample_data['num_words'] = num_words
    sample_data = sample_data[sample_data.num_words > 5]

    hojin_ids = list(set(sample_data.hojin_id))
    hojin_ids = [int(i) for i in hojin_ids]

    # === Tokenizer ===

    tokenizer = Tokenizer(words, data = sample_data, max_len=MAX_LEN)
    
    logger.info("[STEP 3] Preparing web vectors...")

    # === Prepare web vectors ===
    web_vectors = [tokenizer.encode_webportfolio(idx, MAX_PAGE) for idx in tqdm(hojin_ids)]

    # Prepare tensors
    seq_ids = torch.tensor([i[1] for i in web_vectors])
    num_pages = torch.tensor([i[0] for i in web_vectors])
    seq_lengths = tokenizer.max_len - torch.sum(seq_ids == 0, axis=-1)
    labels = torch.tensor([tokenizer.get_label(i) for i in hojin_ids])
    hojin_ids = torch.tensor(hojin_ids)
    
    logger.info(f"seq_ids shape: {seq_ids.shape}, num_pages shape: {num_pages.shape}, seq_lengths shape: {seq_lengths.shape}, labels shape: {labels.shape}, hojin_ids shape: {hojin_ids.shape}")

    # === Create Dataset ===
    logger.info("[STEP 4] Creating dataset...")
    
    dataset = TensorDataset(seq_ids, num_pages, seq_lengths, labels, hojin_ids)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Print number of samples in each set
    logger.info(f"Total samples      : {len(dataset)}")
    logger.info(f"Training samples   : {len(train_set)}")
    logger.info(f"Validation samples : {len(val_set)}")
    logger.info(f"Test samples       : {len(test_set)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,        # Adjust depending on CPU cores
        pin_memory=True       # Required for faster GPU transfers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    vectors = np.array(list(wv_dict.values()))
    words = list(wv_dict.keys())
    vectors_all = np.vstack([np.zeros(300), vectors])

    # === Model ===
    model = DualAttnModel(
        vocab_size=len(words)+1,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        label_dim=LABEL_DIM,
        scale=SCALE,
        page_scale=PAGE_SCALE,
        attn_type=ATTN_TYPE,
        attn_word=ATTN_WORD,
        attn_page=ATTN_PAGE
    )
    
    logger.info(f"Model: {model}")
    
    model.load_vector(pretrained_vectors=vectors_all, trainable=False)
    model = model.to(device)

    logger.info(f"Total params: {count_parameters(model)}")

    # === old code === 
    # # === Training ===
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.02, weight_decay=0.0000, lr_decay=0.01)
    # loss_fn = torch.nn.BCELoss()

    # best_model, history = train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=EPOCHS_DUAL_ATT)
    # torch.save(best_model, DUAL_ATT_SAVE)

    # plot_loss(history)
    # plot_accuracy(history)

    # # === Evaluation ===
    # model.load_state_dict(best_model)
    # model.eval()
    # test_metrics = evaluate(model, test_loader)
    # logger.info(f"Test Metrics: {test_metrics}")
    
    # === new code ===
    # === Training ===
    logger.info("[STEP 6] Starting training...")
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.02, weight_decay=0.0000, lr_decay=0.01)
    loss_fn = torch.nn.BCELoss()

    # Check if model checkpoint exists
    if os.path.exists(DUAL_ATT_SAVE):
        logger.info(f"[STEP 6.1] Loading existing model from {DUAL_ATT_SAVE}...")
        best_model = torch.load(DUAL_ATT_SAVE, map_location=device)
        model.load_state_dict(best_model)
        history = {}  # Empty history since no training is performed
        logger.info("[STEP 6.1] Model loaded successfully, skipping training.")
    else:
        logger.info("[STEP 6.1] No existing model found, starting training...")
        best_model, history = train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=EPOCHS_DUAL_ATT)
        torch.save(best_model, DUAL_ATT_SAVE)
        logger.info(f"[STEP 6.1] Model saved to {DUAL_ATT_SAVE}")

    logger.info("[STEP 6.2] Plotting training metrics...")
    if history:  # Only plot if training was performed
        plot_loss(history)
        plot_accuracy(history)
    else:
        logger.info("[STEP 6.2] No training history available, skipping plotting.")

    # === Evaluation ===
    logger.info("[STEP 7] Evaluating model...")
    model.load_state_dict(best_model)
    model.eval()
    test_metrics = evaluate(model, test_loader)
    logger.info(f"Test Metrics: {test_metrics}")

    # === Embedding + Keyword Extraction ===
    embeddings, selected_df = extract_company_embeddings(model, tokenizer, seq_ids, num_pages, seq_lengths, hojin_ids, sample_data)

    with open(COMPANY_EMBEDDINGS, "wb") as f:
        pickle.dump(embeddings, f)
    selected_df.to_csv(DUAL_ATT_OUTPUT, index=False)
    
    # Group by hojin_id and merge keywords per company
    company_keywords = (
        selected_df
        .groupby("hojin_id")["sents"]
        .apply(lambda x: '|'.join(set('|'.join(x).split('|'))))  # set union of keywords
        .reset_index()
        .rename(columns={"sents": "company_keywords"})
    )

    company_keywords.to_csv(EMBEDDINGS_OUTPUT, index=False)

    logger.info("[DualAttention] Pipeline completed successfully.")

def extract_company_embeddings(model, tokenizer, seq_ids, num_pages, seq_lengths, hojin_ids, data):
    device = next(model.parameters()).device
    model.eval()

    hojin_id_col, url_col, text_col, sents_col, weight_col, hightechflag_col = [], [], [], [], [], []
    company_embeddings = []
    
    sents_selected = []

    for t in tqdm(range(len(hojin_ids)), desc="Processing companies", ncols=100):
        # Running the trained DualAttnModel on each company’s document (a set of webpages) to get:
        # - attn: word-level attention (1 vector per page),
        # - page_attn: attention over pages (page importance),
        # - final_vec: representation of the entire company website,
        # - page_score: raw attention logits for pages.
        with torch.no_grad():
            probs, senti_scores, attn, page_attn, final_vec, page_score, web_vec = model(
                seq_ids[t:t+1].to(device),
                num_pages[t:t+1].to(device),
                seq_lengths[t:t+1].to(device)
            )
        
        id_to_token = tokenizer.id_to_token
        id_to_token[0] = '#'
        
        # Convert token IDs back to words
        # => Transform each page’s word IDs back into real words → forms the page's content as a sentence.
        sents = []
        for i in range(num_pages[t:(t+1)].tolist()[0]):
            sents.append(' '.join([id_to_token[w] for w in seq_ids[t:(t+1)][0][i].tolist()]))

        # For each page with non-zero attention, collect:
        # - URL of the page
        # - Company ID
        # - Original content
        # - Page attention weight
        # - Raw attention score
        company_embeddings.append({
            "hojin_id": int(hojin_ids[t]),
            "embedding": final_vec.detach().cpu().numpy().squeeze()  # shape: (D,)
        })

        n_pages = num_pages[t:(t+1)].item()  # safely convert to Python int
        company_data = data[data.hojin_id == int(hojin_ids[t])]
        
        df = pd.DataFrame({
            'url':           list(company_data.urls),
            'hojin_id':      list(company_data.hojin_id),
            'hightechflag':  list(company_data.hightechflag),
            'text':          list(company_data.cleaned_content),
            'weight':        page_attn.view(-1)[:n_pages].tolist(),
            'page_score':    page_score.view(-1)[:n_pages].tolist(),
            # 'web_vecs':    list(web_vec[0])[:n_pages]
        })
        
        # To get all the websites back together with those with zero weights of attention, we could remove this line of code
        df = df[df.weight > 0].reset_index()

        hojin_id_col.extend(df.hojin_id)
        hightechflag_col.extend(df.hightechflag)
        url_col.extend(df.url)
        text_col.extend(df.text)
        weight_col.extend(df.weight)
        
        # For each page, select keywords based on attention scores
        # For each important page:
        # - Retrieve its words and word-level attention scores,
        # - Select top-N attended words (via percentile threshold),
        # - Save them as the keywords that model considered important.
        for i in list(df['index']):
            sent = sents[i]
            attn1 = attn.squeeze()[i]

            words = sent.split()
            color_array = np.array(attn1.view(-1).tolist())

            selected_keywords, selected_keywords_show = select_keywords(color_array, words, n=20)
            html = colorize(words, selected_keywords_show)

            sents_selected.append([j for j, k in zip(words, selected_keywords_show) if k != 0])
    
    # A list of distinct important words (selected via word-level attention scores).
    sents_selected = ['|'.join(i) for i in sents_selected]
    sents_selected = ['|'.join(list(set(i.split('|')))) for i in sents_selected]

    # Final output contains:
    # - Which pages (URLs) are most relevant (weight),
    # - What words/sentences are important (sents),
    # - Full raw text (text),
    # - Associated company info (hojin_id, hightechflag).
    selected_df = pd.DataFrame({
        'hojin_id': hojin_id_col,
        'url': url_col,
        'weight': weight_col,
        'text':text_col,            # full cleaned text content of a webpage
        'sents': sents_selected,    # important words/sentences from the webpage
        'hightechflag': hightechflag_col,
    })

    return company_embeddings, selected_df




