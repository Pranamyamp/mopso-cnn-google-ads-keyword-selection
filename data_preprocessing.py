"""
MOPSO-Based CNN for Keyword Selection on Google Ads
data_preprocessing.py — Word embedding, corpus selection, class imbalance handling

B.E. Final Year Project, AIT Chikkamagaluru (VTU), 2020-21
"""

import numpy as np
import re
from collections import Counter
from typing import Tuple, List, Dict


# --- Text Cleaning ---

def clean_keyword(text: str) -> str:
    """
    Clean and normalise a keyword string.
    Handles mixed-language keywords (Chinese/English) present in Google Ads data.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def load_and_clean_keywords(filepath: str) -> Tuple[List[str], List[int]]:
    """
    Load keyword dataset from CSV.
    Expected format: keyword, label (0=negative, 1-N=category)

    Returns cleaned keywords and integer labels.
    """
    keywords, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                keyword = clean_keyword(','.join(parts[:-1]))
                label = int(parts[-1])
                keywords.append(keyword)
                labels.append(label)
    return keywords, labels


# --- Vocabulary & Tokenisation ---

def build_vocabulary(keywords: List[str], min_freq: int = 1) -> Tuple[Dict, Dict]:
    """
    Build word-to-index vocabulary from keyword corpus.

    Returns:
        vocab: word → index mapping
        inv_vocab: index → word mapping
    """
    word_counts = Counter()
    for kw in keywords:
        word_counts.update(kw.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def pad_sentences(sequences: List[List[int]], max_len: int = 50) -> np.ndarray:
    """
    Pad or truncate sequences to fixed length.
    Shorter sequences are zero-padded on the right.
    """
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded


def encode_keywords(keywords: List[str], vocab: Dict, max_len: int = 50) -> np.ndarray:
    """Convert keyword strings to padded integer sequences."""
    sequences = [
        [vocab.get(word, vocab['<UNK>']) for word in kw.split()]
        for kw in keywords
    ]
    return pad_sentences(sequences, max_len)


# --- GloVe Word Embeddings ---

def load_glove_embeddings(glove_path: str, embedding_dim: int = 100) -> Dict[str, np.ndarray]:
    """
    Load pretrained GloVe word vectors.
    Download from: https://nlp.stanford.edu/projects/glove/

    GloVe was chosen over Word2Vec and FastText for this project
    based on performance on the multilingual keyword corpus.
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    print(f"Loaded {len(embeddings):,} GloVe vectors (dim={embedding_dim})")
    return embeddings


def build_embedding_matrix(
    vocab: Dict,
    glove_embeddings: Dict,
    embedding_dim: int = 100
) -> np.ndarray:
    """
    Build embedding weight matrix for Keras Embedding layer.
    Words not in GloVe are initialised with small random values.
    """
    matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim)).astype(np.float32)
    matrix[0] = np.zeros(embedding_dim)  # PAD token = zero vector

    found = 0
    for word, idx in vocab.items():
        if word in glove_embeddings:
            matrix[idx] = glove_embeddings[word]
            found += 1

    print(f"Embedding matrix: {found}/{len(vocab)} words found in GloVe")
    return matrix


# --- Class Imbalance: SMOTE ---

def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbours: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic Minority Oversampling Technique (SMOTE).

    The Google Ads keyword dataset is heavily imbalanced — very few keywords
    are truly relevant to an advertiser. SMOTE generates synthetic samples
    for minority classes by interpolating between existing samples.

    For production use, prefer: from imblearn.over_sampling import SMOTE
    This implementation demonstrates the core concept.
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = [X]
    y_resampled = [y]

    for cls, count in zip(classes, counts):
        if count == max_count:
            continue

        X_cls = X[y == cls]
        n_synthetic = max_count - count

        synthetic = []
        for _ in range(n_synthetic):
            # Pick a random sample from minority class
            idx = np.random.randint(0, len(X_cls))
            sample = X_cls[idx]

            # Pick a random neighbour
            neighbour_idx = np.random.randint(0, len(X_cls))
            neighbour = X_cls[neighbour_idx]

            # Interpolate
            lam = np.random.uniform(0, 1)
            new_sample = sample + lam * (neighbour - sample)
            synthetic.append(new_sample.astype(np.int32))

        X_resampled.append(np.array(synthetic))
        y_resampled.append(np.full(n_synthetic, cls, dtype=y.dtype))

    X_out = np.vstack(X_resampled)
    y_out = np.concatenate(y_resampled)

    # Shuffle
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# --- Train/Val Split ---

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple:
    """Split data into train and validation sets."""
    np.random.seed(seed)
    n = len(y)
    perm = np.random.permutation(n)
    split = int(n * (1 - val_ratio))

    train_idx = perm[:split]
    val_idx = perm[split:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


if __name__ == '__main__':
    # Demo with synthetic data
    print("=== Data Preprocessing Demo ===")

    sample_keywords = [
        "buy electronics online", "digital electronics seller",
        "cheap electronics store", "best buy electronics near me",
        "cosmetics shop online", "buy makeup products",
        "furniture store near me", "discount furniture",
        "clothing sale online", "fashion store"
    ]
    sample_labels = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]

    cleaned = [clean_keyword(kw) for kw in sample_keywords]
    vocab, inv_vocab = build_vocabulary(cleaned)
    print(f"Vocabulary size: {len(vocab)}")

    encoded = encode_keywords(cleaned, vocab, max_len=10)
    print(f"Encoded shape: {encoded.shape}")

    X_res, y_res = smote_oversample(encoded, np.array(sample_labels))
    print(f"After SMOTE: {X_res.shape[0]} samples (was {len(sample_labels)})")
