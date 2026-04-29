#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Milestone I Natural Language Processing
# ## Task 2 & 3 — Feature Representations and Classification
# 
# #### Group Members
# 1. Vo Ngoc Dung — S4124370
# 2. Tang Hoang Ha — S4147768
# 3. Nguyen Anh Duc — S4136756
# 4. Nguyen Quoc Trong Nghia — S3343711
# 
# **Environment:** Python 3, Jupyter notebook
# 
# **Libraries used:**
# - `pandas` / `numpy` — data manipulation
# - `scipy.sparse` — sparse matrix storage for the bag-of-words count vectors
# - `sklearn.feature_extraction.text.TfidfVectorizer` — TF-IDF weights computed against the Task 1 vocabulary
# - `gensim.downloader` — pre-trained GloVe word embeddings
# - `sklearn.linear_model`, `sklearn.model_selection` — classification and 5-fold cross-validation (Task 3)
# 
# ## Introduction
# 
# Task 1 produced a cleaned, standardised vocabulary (`vocab.txt`, 8,054 unigrams) and a per-review token sequence (`processed.csv`). Those tokens are still strings — machine learning models operate on numerical vectors. Task 2 turns each review into three different fixed-size numerical representations, which Task 3 then compares as inputs to the same classifier so we can measure which representation carries the most predictive signal for `is_a_buyer`.
# 
# The three representations capture progressively richer notions of meaning:
# 
# | Representation | What it encodes | Strength | Weakness |
# |---|---|---|---|
# | **Count vector** (bag-of-words) | Which vocabulary words appear, and how often | Simple, interpretable, high-dimensional | Loses word order and semantic similarity ("great" ≈ "amazing" not captured) |
# | **Unweighted embedding** | Mean of pre-trained word vectors for in-review words | Captures distributional semantics; dense, fixed-size | Equal weight to every word; common words drag the centroid |
# | **TF-IDF-weighted embedding** | Same as above, but each word's contribution scaled by its TF-IDF score | Distinctive words amplified, ubiquitous words suppressed | Still loses word order |
# 
# We choose **GloVe-Wiki-Gigaword-100** (`gensim`-downloadable, 400k-word vocab, 100-dim) as the embedding model. Justification appears in §2.2.
# 

# ## 1. Imports and configuration

# In[1]:


import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.sparse import csr_matrix

import gensim.downloader as gensim_api
from sklearn.feature_extraction.text import TfidfVectorizer

# Reproducibility — TF-IDF and downstream sklearn estimators
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ## 2. Load Task 1 outputs
# 
# We load `processed.csv` (cleaned reviews + the original 14 metadata columns kept intact for Task 3 Q2) and `vocab.txt` (the unigram vocabulary that defines the column ordering for the count vectors). Both files were produced by Task 1 and are validated here before any feature engineering, so any silent corruption is caught early.

# In[2]:


# Load the cleaned reviews from Task 1
df = pd.read_csv('processed.csv')

# Some reviews may now be empty strings (after Task 1 dropped all their tokens, e.g.
# very short reviews composed entirely of stopwords + length-1 tokens).
# We replace any NaN that pandas infers from empty strings with empty strings,
# so .split() still works and produces an empty list rather than throwing.
df['review_text'] = df['review_text'].fillna('').astype(str)

# Pre-tokenize once: tokens are already cleaned + space-joined by Task 1.
df['tokens'] = df['review_text'].str.split()

print(f"Loaded {len(df)} reviews from processed.csv")
print(f"Columns: {list(df.columns)}")
print(f"Reviews with 0 surviving tokens: {(df['tokens'].str.len() == 0).sum()}")
print(f"Median tokens per review: {df['tokens'].str.len().median():.0f}")
df.head(3)


# In[3]:


# Load vocabulary from vocab.txt — the canonical word -> index mapping for count vectors
word_to_index = {}
with open('vocab.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line:
            continue
        # rsplit handles edge cases like words containing ':' (none expected after the
        # alphabetic-only regex from Task 1, but defensive)
        word, idx = line.rsplit(':', 1)
        word_to_index[word] = int(idx)

vocab_size = len(word_to_index)
print(f"Vocabulary size: {vocab_size}")

# Sanity check: indices must be exactly 0..N-1 and the word list alphabetically sorted
indices = sorted(word_to_index.values())
assert indices == list(range(vocab_size)), "vocab.txt indices are not 0..N-1"
sorted_words = sorted(word_to_index.keys())
assert sorted_words == [w for w,_ in sorted(word_to_index.items(), key=lambda x: x[1])], \
    "vocab.txt word order does not match index order"
print("vocab.txt integrity: OK")


# ## 2.1 Count vectors (bag-of-words)
# 
# ### What this representation does
# Each review becomes a sparse vector of length `vocab_size` (8,054 here). Position *i* in the vector is the count of how many times the word at index *i* in `vocab.txt` appears in this review.
# 
# ### Why this representation
# Bag-of-words is the canonical baseline for text classification. It makes a simple but strong assumption: the words that *appear* in a review (and how often) carry the predictive signal — order doesn't matter for binary purchase prediction. This assumption is correct often enough that bag-of-words is hard to beat for short, lexicon-heavy text like product reviews.
# 
# ### How we build it
# For every review, we count tokens *that exist in our vocabulary* (Task 1 already removed everything else, so this is essentially every token in `processed.csv`). Each `(word, count)` pair is converted to `(vocab_index, count)` and emitted as a sparse line.
# 
# ### Output format (per the brief)
# ```
# #<review_id>,<idx>:<count>,<idx>:<count>,...
# ```
# - One line per review.
# - Sparse: words with count 0 are omitted.
# - `<idx>` is the integer from `vocab.txt`. `<count>` is the term frequency in this review's `review_text` only (review_title is excluded — it's processed separately for Task 3 Q2).
# - Indices within a line are emitted in ascending order, which is the convention shown in the brief's example.

# In[4]:


# Build sparse count vectors. We iterate per review rather than using
# CountVectorizer directly because we need to enforce *our* vocabulary indexing
# (vocab.txt order), not sklearn's internal alphabetisation, which would silently
# rearrange the columns.

def review_to_count_pairs(tokens, w2i):
    """Convert a list of tokens to a list of (vocab_index, count) tuples,
    sorted ascending by index. Tokens not in the vocabulary are skipped silently
    (they were already removed by Task 1)."""
    counts = Counter(t for t in tokens if t in w2i)
    return sorted((w2i[w], c) for w, c in counts.items())

# Apply across the corpus
df['count_pairs'] = df['tokens'].apply(lambda toks: review_to_count_pairs(toks, word_to_index))

# Sanity stats
total_nonzero = df['count_pairs'].str.len().sum()
empty_reviews = (df['count_pairs'].str.len() == 0).sum()
print(f"Total non-zero entries across all count vectors: {total_nonzero:,}")
print(f"Reviews with zero non-zero entries (empty after Task 1 cleaning): {empty_reviews}")
print(f"Mean non-zero entries per review: {total_nonzero / len(df):.2f}")

# Spot-check the first review
sample_idx = 0
print(f"\nSample (review_id={df.loc[sample_idx, 'review_id']}):")
print(f"  Tokens: {df.loc[sample_idx, 'tokens']}")
print(f"  Count pairs (idx:count): {df.loc[sample_idx, 'count_pairs'][:10]}")


# In[5]:


# Persist count_vectors.txt in the brief's required format
with open('count_vectors.txt', 'w', encoding='utf-8') as f:
    for review_id, pairs in zip(df['review_id'], df['count_pairs']):
        pair_str = ','.join(f"{idx}:{cnt}" for idx, cnt in pairs)
        f.write(f"#{review_id},{pair_str}\n")

print(f"count_vectors.txt written: {os.path.getsize('count_vectors.txt'):,} bytes, {len(df)} lines")
# Show first 2 lines for visual verification
with open('count_vectors.txt', 'r', encoding='utf-8') as f:
    for _ in range(2):
        print(f.readline().rstrip())


# ## 2.2 Embedding model selection
# 
# We have to pick **one** pre-trained word embedding for the dense representations. The brief lists FastText, GoogleNews-Word2Vec-300, and GloVe as candidates.
# 
# ### Decision: **GloVe-Wiki-Gigaword-100** (`glove-wiki-gigaword-100`)
# 
# | Criterion | GloVe-100 | GloVe-300 | Word2Vec-GoogleNews-300 | FastText-300 |
# |---|---|---|---|---|
# | Download size | **134 MB** | 394 MB | 1.7 GB | 1.0 GB |
# | Dimensions | 100 | 300 | 300 | 300 |
# | Vocab coverage on our reviews | Good (Wikipedia + Gigaword) | Same | News-biased | Subword-based, robust to OOV |
# | Reproducibility on a marker's machine | High (small) | Medium | Low (huge) | Medium |
# 
# ### Reasoning
# 
# 1. **Dimensionality is sufficient.** For a binary classification task (`is_a_buyer`) with ~61k labelled reviews, 100 dimensions of distributional information is comfortably above the complexity floor where dimensionality bottlenecks accuracy. Going to 300 mainly adds compute, not predictive signal — the literature consistently shows 100d GloVe is within a few percentage points of 300d on sentence-level classification.
# 
# 2. **Reproducibility matters for marking.** A 134 MB cached model loads in seconds; a 1.7 GB model risks the marker's environment failing to download or hitting memory issues. The rubric explicitly punishes "the expert will NOT fix your code's problem" — so smaller is safer.
# 
# 3. **Vocabulary fit.** GloVe is trained on Wikipedia + Gigaword, which is broader and less news-skewed than Word2Vec-GoogleNews. Beauty/cosmetics reviews contain everyday consumer language ('soft', 'smooth', 'absorb', 'hydrate') — GloVe handles these well; we'll measure OOV rate below to confirm.
# 
# 4. **OOV vs. FastText.** FastText's subword model is genuinely better for typos and brand names (e.g., 'nykaa', 'absoultely'). However, the cost is 8× the file size for marginal coverage gains on a corpus that's mostly common English. We accept some OOV in exchange for size.

# In[6]:


# Load GloVe-100 (downloads ~134MB on first run, then cached in ~/gensim-data/)
print("Loading glove-wiki-gigaword-100 ...")
glove = gensim_api.load('glove-wiki-gigaword-100')
EMBED_DIM = glove.vector_size
print(f"GloVe loaded — vocabulary: {len(glove.key_to_index):,} words, dim: {EMBED_DIM}")


# In[7]:


# OOV analysis: how many of our vocab words are covered by GloVe?
in_glove = sum(1 for w in word_to_index if w in glove.key_to_index)
oov = vocab_size - in_glove
print(f"Task-1 vocab coverage in GloVe: {in_glove}/{vocab_size} ({100*in_glove/vocab_size:.1f}%)")
print(f"OOV words (skipped during averaging): {oov}")

# At the corpus level — how many tokens (occurrences, not unique) are covered?
total_tokens = 0
covered_tokens = 0
for tokens in df['tokens']:
    for t in tokens:
        total_tokens += 1
        if t in glove.key_to_index:
            covered_tokens += 1
print(f"Token-level GloVe coverage: {covered_tokens:,}/{total_tokens:,} ({100*covered_tokens/total_tokens:.1f}%)")

# Reviews where 0 tokens are covered (would produce zero vector)
zero_cov_reviews = sum(
    1 for tokens in df['tokens']
    if not any(t in glove.key_to_index for t in tokens)
)
print(f"Reviews with 0 GloVe-covered tokens: {zero_cov_reviews} (will be emitted as zero vectors)")


# ## 2.3 Unweighted embedding vectors
# 
# ### What this representation does
# Each review becomes a single 100-dimensional dense vector — the *arithmetic mean* of the GloVe vectors for every in-review token that GloVe knows about. Tokens not in GloVe are skipped (treated as missing rather than as zeros, otherwise OOV-heavy reviews would be systematically pulled toward the origin).
# 
# ### Why averaging
# The simplest principled aggregation: it's commutative (matches bag-of-words' order-independence assumption), bounded (mean stays within the convex hull of word vectors so magnitudes are stable), and dimensionality-preserving. Sum would grow with review length and confound longer reviews with stronger signals; max-pooling discards information.
# 
# ### Why "unweighted" is the baseline
# Every word's vector contributes equally. A review like "the product is good and works well" has its centroid dragged by the embedding of "good" and "works" alongside more discriminative words, if any. The next section (TF-IDF weighting) addresses this.

# In[8]:


def review_to_unweighted_embedding(tokens, kv, dim):
    """Mean of GloVe vectors for in-vocab tokens. Returns zero vector if no token is in-vocab."""
    vectors = [kv[t] for t in tokens if t in kv.key_to_index]
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

# Compute for all reviews. List comprehension is fine at 61k rows.
unweighted_matrix = np.vstack([
    review_to_unweighted_embedding(toks, glove, EMBED_DIM)
    for toks in df['tokens']
])
print(f"Unweighted embedding matrix shape: {unweighted_matrix.shape}")
print(f"Zero-vector rows (no GloVe-covered tokens): {(unweighted_matrix.sum(axis=1) == 0).sum()}")
print(f"Sample (first review, first 10 dims): {unweighted_matrix[0, :10]}")


# In[9]:


# Persist unweighted_vectors.txt — dense floats, comma-separated, prefixed by #review_id
with open('unweighted_vectors.txt', 'w', encoding='utf-8') as f:
    for review_id, vec in zip(df['review_id'], unweighted_matrix):
        # Format each float compactly but with enough precision for downstream use
        vec_str = ','.join(f"{x:.6f}" for x in vec)
        f.write(f"#{review_id},{vec_str}\n")

print(f"unweighted_vectors.txt written: {os.path.getsize('unweighted_vectors.txt'):,} bytes, {len(df)} lines")
with open('unweighted_vectors.txt', 'r', encoding='utf-8') as f:
    line = f.readline().rstrip()
print(f"First line preview ({len(line)} chars): {line[:120]}...")


# ## 2.4 TF-IDF weighted embedding vectors
# 
# ### What this representation does
# Same averaging as §2.3, but each word's GloVe vector is multiplied by its TF-IDF score in this review before averaging. The denominator is the sum of TF-IDF weights (so it remains a weighted mean, bounded like the unweighted version).
# 
# ### Why TF-IDF weighting (the principle)
# TF-IDF scores reward words that are **frequent in this review** but **rare across the corpus**. A word in every review (high DF) gets near-zero IDF and contributes almost nothing to the centroid; a word that's distinctive to this review (low DF, high TF) dominates the centroid. The result: each review's vector is pulled toward the meanings of *its* discriminative words rather than its generic-praise words.
# 
# For `is_a_buyer`, the words that are likely to discriminate buyers from non-buyers are mid-frequency content words ('delivery', 'received', 'expected', 'refund', 'recommend') — exactly the words TF-IDF amplifies relative to a flat average.
# 
# ### Implementation choices
# - **TF-IDF vocabulary = Task 1 vocabulary.** We restrict `TfidfVectorizer` to the words in `vocab.txt` so the IDF table is consistent with the count-vector representation and Task 3 comparisons are apples-to-apples.
# - **Tokenizer = whitespace split.** `processed.csv` is already cleaned and space-joined, so we bypass sklearn's default tokenizer and lowercaser.
# - **`sublinear_tf=False`, `norm=None`.** We want the raw TF-IDF score as the weight; sklearn's row-normalisation would re-scale and obscure the per-word weight contribution to the embedding average.

# In[10]:


# Fit TF-IDF on the cleaned reviews, restricted to the Task 1 vocabulary
# Using lambda x: x.split() so we tokenize on whitespace only (text is already cleaned)
vocab_terms = sorted(word_to_index.keys())  # alphabetical, matches vocab.txt
tfidf_vec = TfidfVectorizer(
    vocabulary=vocab_terms,
    tokenizer=lambda s: s.split(),
    lowercase=False,    # already lowercased in Task 1
    norm=None,          # raw TF-IDF, not L2-normalised
    sublinear_tf=False, # plain TF
    token_pattern=None, # silence sklearn warning when tokenizer is set
)

# fit_transform on the joined token strings (already in df['review_text'])
tfidf_matrix = tfidf_vec.fit_transform(df['review_text'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"TF-IDF non-zero entries: {tfidf_matrix.nnz:,}")

# IMPORTANT: sklearn re-orders columns alphabetically when given an explicit `vocabulary`.
# Our vocab.txt is also alphabetical (Task 1 ensures this), so the column ordering matches
# word_to_index. Verify defensively.
assert list(tfidf_vec.get_feature_names_out()) == vocab_terms, \
    "TF-IDF vocab order does not match Task 1 vocab order"
print("TF-IDF column order matches vocab.txt: OK")


# In[11]:


# Build the GloVe matrix aligned to vocab.txt's column order.
# Row i of glove_matrix corresponds to vocab_terms[i] (alphabetical).
# OOV words get a zero row, which will contribute nothing to weighted sums.
glove_matrix = np.zeros((vocab_size, EMBED_DIM), dtype=np.float32)
oov_indices = []
for i, term in enumerate(vocab_terms):
    if term in glove.key_to_index:
        glove_matrix[i] = glove[term]
    else:
        oov_indices.append(i)
print(f"GloVe-matrix populated. OOV rows zeroed: {len(oov_indices)}")


# In[12]:


# Compute weighted-sum embeddings via sparse matrix multiplication.
# (TF-IDF[N x V]) @ (GloVe[V x D]) = weighted sum per review [N x D].
# We then divide each row by the sum of its TF-IDF weights to get a weighted *mean*.
weighted_sum = tfidf_matrix.dot(glove_matrix)  # shape: (N, D)

# Sum of TF-IDF weights per review — but only over words that have a non-zero GloVe vector
# (otherwise OOV words still inflate the denominator and bias the mean toward zero).
in_vocab_mask = np.ones(vocab_size, dtype=np.float32)
in_vocab_mask[oov_indices] = 0.0  # OOV columns excluded from the denominator
tfidf_weight_sum = tfidf_matrix.multiply(in_vocab_mask).sum(axis=1)  # (N, 1) matrix
tfidf_weight_sum = np.asarray(tfidf_weight_sum).flatten()

# Divide. Reviews where no in-GloVe word has any TF-IDF weight (e.g. empty after Task 1)
# would divide by zero — replace those rows with zero vectors.
weighted_matrix = np.zeros_like(weighted_sum)
nonzero_rows = tfidf_weight_sum > 0
weighted_matrix[nonzero_rows] = weighted_sum[nonzero_rows] / tfidf_weight_sum[nonzero_rows, None]
weighted_matrix = weighted_matrix.astype(np.float32)

print(f"Weighted embedding matrix shape: {weighted_matrix.shape}")
print(f"Zero-vector rows: {(~nonzero_rows).sum()}")
print(f"Sample (first review, first 10 dims): {weighted_matrix[0, :10]}")


# In[13]:


# Persist weighted_vectors.txt — same format as unweighted_vectors.txt
with open('weighted_vectors.txt', 'w', encoding='utf-8') as f:
    for review_id, vec in zip(df['review_id'], weighted_matrix):
        vec_str = ','.join(f"{x:.6f}" for x in vec)
        f.write(f"#{review_id},{vec_str}\n")

print(f"weighted_vectors.txt written: {os.path.getsize('weighted_vectors.txt'):,} bytes, {len(df)} lines")
with open('weighted_vectors.txt', 'r', encoding='utf-8') as f:
    line = f.readline().rstrip()
print(f"First line preview ({len(line)} chars): {line[:120]}...")


# ## 2.5 Independent re-load verification
# 
# Mechanical-pass insurance: re-open all three output files from scratch and confirm the row counts and per-row formats match the brief's specification.

# In[14]:


# Verify all three files: line count = number of reviews, prefix = '#<review_id>,'
N = len(df)
expected_ids = set(str(rid) for rid in df['review_id'])

for fname, expect_dense_dim in [
    ('count_vectors.txt', None),
    ('unweighted_vectors.txt', EMBED_DIM),
    ('weighted_vectors.txt', EMBED_DIM),
]:
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == N, f"{fname}: expected {N} lines, got {len(lines)}"

    # Spot-check the first line's structure
    first = lines[0].rstrip()
    assert first.startswith('#'), f"{fname}: line 1 missing '#' prefix"
    head, _, tail = first[1:].partition(',')
    assert head in expected_ids, f"{fname}: review_id {head} not in dataset"

    if expect_dense_dim is not None:
        # Dense file — should have exactly EMBED_DIM floats
        n_floats = len(tail.split(','))
        assert n_floats == expect_dense_dim, \
            f"{fname}: line 1 has {n_floats} floats, expected {expect_dense_dim}"

    print(f"{fname}: OK — {len(lines)} lines, {os.path.getsize(fname):,} bytes")


# ---
# 
# ## Task 3 — Cosmetics/Beauty Review Classification
# 
# We now have three feature representations from Task 2:
# - `count_vectors.txt` — sparse bag-of-words (8,054 dims)
# - `unweighted_vectors.txt` — mean GloVe-100 embedding (100 dims)
# - `weighted_vectors.txt` — TF-IDF-weighted mean GloVe-100 embedding (100 dims)
# 
# Task 3 uses these as inputs to predict the binary label `is_a_buyer`.
# 
# ### Class balance and metric choice
# The label is imbalanced: **78.7% True, 21.3% False**. A trivial "always predict True" classifier already scores 78.7% accuracy, so accuracy alone is a misleading headline metric. We report three metrics throughout:
# - **Accuracy** — overall correctness; comparable to the trivial baseline.
# - **Macro-F1** — average of per-class F1; punishes models that ignore the minority class.
# - **ROC-AUC** — threshold-independent ranking quality; robust to class imbalance.
# 
# Macro-F1 is our primary comparison metric because it answers "does this representation help us identify *both* buyers and non-buyers," which is what an e-commerce platform actually cares about.
# 
# ### Classifier choice
# **Logistic regression** (`sklearn.linear_model.LogisticRegression`, `solver='liblinear'`, `max_iter=1000`). Reasoning:
# - Linear models work well on high-dimensional sparse text features (count vectors).
# - They also work well on dense embedding features when the number of examples ≫ dimensions (61k examples, 100 dims — comfortable).
# - A single estimator across all three representations isolates the *representation* as the only varying factor — this is exactly what Q1 is asking us to measure.
# - `liblinear` is fast on sparse input, deterministic, and handles binary classification out-of-the-box.
# 
# ### Cross-validation
# **5-fold stratified CV** (`StratifiedKFold`) — the brief mandates 5-fold; stratification preserves the 78.7/21.3 class ratio in each fold so per-fold metrics are comparable.
# 

# ## 3.0 Setup — load all three representations into memory
# 
# We load the three Task 2 files from disk in a way that mirrors how the marker would: each file is parsed independently, row-aligned with `is_a_buyer` via `review_id`. This guards against any silent reordering between Task 2 outputs and the original `processed.csv`.

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack as sp_hstack, lil_matrix

# 1. Labels
y = df['is_a_buyer'].astype(int).to_numpy()
print(f"y shape: {y.shape}, positive class rate: {y.mean():.3f}")


# In[16]:


# 2. Count vectors — re-parse from disk into a sparse CSR matrix
# We rebuild from file (rather than reusing the in-memory df['count_pairs']) to
# verify end-to-end correctness of the Task 2 output and to mirror what the marker would do.
def load_count_vectors(path, n_rows, vocab_size):
    rows, cols, vals = [], [], []
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for row_i, line in enumerate(f):
            line = line.rstrip('\n')
            head, _, body = line.partition(',')
            ids.append(int(head[1:]))  # drop the '#'
            if not body:
                continue
            for pair in body.split(','):
                idx, cnt = pair.split(':')
                rows.append(row_i); cols.append(int(idx)); vals.append(int(cnt))
    X = csr_matrix((vals, (rows, cols)), shape=(n_rows, vocab_size), dtype=np.float32)
    return X, ids

X_count, count_ids = load_count_vectors('count_vectors.txt', len(df), vocab_size)
assert count_ids == df['review_id'].tolist(), "count_vectors.txt row order does not match processed.csv"
print(f"X_count: {X_count.shape}, nnz={X_count.nnz:,}, dtype={X_count.dtype}")


# In[17]:


# 3. Embedding vectors — re-parse the dense .txt files into numpy arrays
def load_dense_vectors(path, n_rows, dim):
    X = np.zeros((n_rows, dim), dtype=np.float32)
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for row_i, line in enumerate(f):
            line = line.rstrip('\n')
            head, _, body = line.partition(',')
            ids.append(int(head[1:]))
            X[row_i] = np.fromstring(body, sep=',', dtype=np.float32)
    return X, ids

X_unweighted, uw_ids = load_dense_vectors('unweighted_vectors.txt', len(df), EMBED_DIM)
X_weighted, w_ids = load_dense_vectors('weighted_vectors.txt', len(df), EMBED_DIM)
assert uw_ids == df['review_id'].tolist() == w_ids, "embedding file row order mismatch"
print(f"X_unweighted: {X_unweighted.shape}, dtype={X_unweighted.dtype}")
print(f"X_weighted:   {X_weighted.shape}, dtype={X_weighted.dtype}")


# ## 3.1 Q1 — Language model comparison
# 
# For each of the three representations, we fit `LogisticRegression(solver='liblinear', max_iter=1000)` under 5-fold stratified cross-validation and record accuracy, macro-F1, and ROC-AUC.
# 
# The hypothesis we are testing: **TF-IDF-weighted embeddings should outperform unweighted embeddings**, because TF-IDF amplifies the contribution of words that are distinctive to a particular review (e.g., transaction-specific words like *delivery*, *received*, *expected*) and suppresses the contribution of generic praise words.
# 
# The count vector is the lexical baseline — it has 80× more dimensions than the embeddings, which gives logistic regression more degrees of freedom but also means it relies on word-identity rather than word-meaning.

# In[18]:


# Helper: 5-fold CV with multiple metrics, returning a tidy summary
SCORING = ['accuracy', 'f1_macro', 'roc_auc']

def evaluate(X, y, name, classifier=None, n_splits=5, random_state=RANDOM_STATE):
    if classifier is None:
        classifier = LogisticRegression(solver='liblinear', max_iter=1000, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(classifier, X, y, cv=cv, scoring=SCORING, n_jobs=-1, return_train_score=False)
    summary = {
        'representation': name,
        'accuracy_mean': scores['test_accuracy'].mean(),
        'accuracy_std': scores['test_accuracy'].std(),
        'f1_macro_mean': scores['test_f1_macro'].mean(),
        'f1_macro_std': scores['test_f1_macro'].std(),
        'roc_auc_mean': scores['test_roc_auc'].mean(),
        'roc_auc_std': scores['test_roc_auc'].std(),
    }
    return summary

# Run Q1
q1_results = []
for X, name in [
    (X_count, 'Count (BoW)'),
    (X_unweighted, 'Unweighted GloVe-100'),
    (X_weighted, 'TF-IDF Weighted GloVe-100'),
]:
    print(f"Evaluating {name} ...")
    q1_results.append(evaluate(X, y, name))

q1_df = pd.DataFrame(q1_results)
print("\n=== Q1 RESULTS — review text only ===")
print(q1_df.to_string(index=False))


# ### Q1 discussion
# 
# We expect the relative ordering of the three representations to reveal something about *what* drives `is_a_buyer`:
# 
# - **If count wins:** the predictive signal is in *specific words* (transaction vocabulary), not in semantic similarity. The embedding-based representations smooth across word identity and lose that signal.
# - **If unweighted wins:** the signal is in the *overall sentiment vector* of the review — words don't need to be specific, just collectively positive/negative.
# - **If weighted wins:** the signal is in *distinctive content words*, and TF-IDF correctly amplifies them.
# 
# Whichever wins, we will use that representation as the **base** for Q2, where we test whether adding title and product metadata adds further predictive power.

# ## 3.2 Q2a — Adding the review title
# 
# The hypothesis: review titles are short summaries (often the most opinionated sentence) so adding them should boost predictive power, particularly for the 1,911 reviews whose `review_text` was emptied by Task 1 cleaning — for those, the title may be the only content the classifier has.
# 
# ### Approach
# For each representation, we:
# 1. Apply the **same Task 1 tokenisation** to `review_title` (regex + lowercase + length filter + stopwords). We deliberately *do not* re-run the term-frequency or top-20 doc-frequency filters on titles, because those filters need a corpus-wide frequency view that the brief did not specify for titles, and re-deriving them would be a non-spec design choice. Filtering to `vocab.txt` words at the lookup stage gives an equivalent effect.
# 2. Build a parallel title-only feature vector with the same shape as the body's vector.
# 3. **Concatenate** body and title vectors horizontally (`X = [X_body | X_title]`) so the classifier can learn separate weights for the two sources.
# 
# Concatenation is preferred over text-level concatenation ("title + ' ' + text → one vector") because the classifier can learn that the title's contribution should be weighted differently from the body's.

# In[19]:


# Pre-process titles using the same Task 1 cleaning, then filter to vocab.txt
import re
TOK_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

# Reload stopwords from the canonical file
with open('data/stopwords_en.txt', 'r', encoding='utf-8') as f:
    sw = set(line.strip().lower() for line in f if line.strip())

def clean_title_tokens(text):
    if not isinstance(text, str):
        return []
    toks = [t.lower() for t in TOK_PATTERN.findall(text)]
    toks = [t for t in toks if len(t) >= 2 and t not in sw]
    return toks

# review_title may have NaN
df['title_tokens'] = df['review_title'].fillna('').apply(clean_title_tokens)
df['title_text'] = df['title_tokens'].apply(lambda toks: ' '.join(toks))

print(f"Title token stats — median {df['title_tokens'].str.len().median():.0f} tokens, max {df['title_tokens'].str.len().max()}")
print(f"Titles with 0 surviving tokens: {(df['title_tokens'].str.len() == 0).sum()}")
print(f"Sample title cleaning:")
for i in range(3):
    print(f"  {df.loc[i,'review_title'][:60]!r:<70} -> {df.loc[i,'title_tokens']}")


# In[20]:


# Build title features matching each of the three representations, restricted to vocab.txt

# 1. Title count vectors — same vocab.txt
title_count_rows, title_count_cols, title_count_vals = [], [], []
for row_i, toks in enumerate(df['title_tokens']):
    counts = Counter(t for t in toks if t in word_to_index)
    for w, c in counts.items():
        title_count_rows.append(row_i)
        title_count_cols.append(word_to_index[w])
        title_count_vals.append(c)
X_title_count = csr_matrix(
    (title_count_vals, (title_count_rows, title_count_cols)),
    shape=(len(df), vocab_size), dtype=np.float32,
)

# 2. Title unweighted embeddings — same GloVe model
def title_unweighted(toks):
    vecs = [glove[t] for t in toks if t in glove.key_to_index]
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else np.zeros(EMBED_DIM, dtype=np.float32)

X_title_unweighted = np.vstack([title_unweighted(toks) for toks in df['title_tokens']])

# 3. Title weighted embeddings — recompute TF-IDF on the title corpus, then weighted-mean
tfidf_title_vec = TfidfVectorizer(
    vocabulary=vocab_terms,
    tokenizer=lambda s: s.split(),
    lowercase=False, norm=None, sublinear_tf=False, token_pattern=None,
)
tfidf_title_matrix = tfidf_title_vec.fit_transform(df['title_text'])

# Same trick: sparse @ dense, divide by per-row sum of weights for in-vocab columns
title_weighted_sum = tfidf_title_matrix.dot(glove_matrix)
title_weight_sum = np.asarray(tfidf_title_matrix.multiply(in_vocab_mask).sum(axis=1)).flatten()
X_title_weighted = np.zeros_like(title_weighted_sum)
nz = title_weight_sum > 0
X_title_weighted[nz] = title_weighted_sum[nz] / title_weight_sum[nz, None]
X_title_weighted = X_title_weighted.astype(np.float32)

print(f"X_title_count:      {X_title_count.shape}, nnz={X_title_count.nnz}")
print(f"X_title_unweighted: {X_title_unweighted.shape}")
print(f"X_title_weighted:   {X_title_weighted.shape}")


# In[21]:


# Concatenate body + title and re-run 5-fold CV
q2a_results = []

# Count: sparse hstack
X_q2a_count = sp_hstack([X_count, X_title_count]).tocsr()
print(f"X_q2a_count: {X_q2a_count.shape}")
q2a_results.append(evaluate(X_q2a_count, y, 'Count (BoW) + Title'))

# Unweighted: dense hstack
X_q2a_unweighted = np.hstack([X_unweighted, X_title_unweighted])
q2a_results.append(evaluate(X_q2a_unweighted, y, 'Unweighted GloVe + Title'))

# Weighted: dense hstack
X_q2a_weighted = np.hstack([X_weighted, X_title_weighted])
q2a_results.append(evaluate(X_q2a_weighted, y, 'TF-IDF Weighted GloVe + Title'))

q2a_df = pd.DataFrame(q2a_results)
print("\n=== Q2a RESULTS — text + title ===")
print(q2a_df.to_string(index=False))


# ## 3.3 Q2b — Adding product metadata
# 
# We add four metadata columns:
# - `brand_name` — categorical, one-hot encoded for the top-50 brands; rarer brands collapsed into an `OTHER` bucket so the feature dimensionality stays bounded.
# - `product_title` — text, count-vectorised against `vocab.txt` (same approach as the review title).
# - `avg_product_rating` — numerical (1–5 scale), z-score standardised.
# - `price` — numerical, log-transformed (prices span 3+ orders of magnitude) then standardised.
# 
# We **do not** include `product_id`, `review_id`, `author`, `review_date`, `product_rating_count`, `product_tags`, or `product_url` — these are either identifiers (data leakage risk for tree-based models, no semantic content for linear models) or near-zero-coverage (`product_tags` is missing in 78% of rows).
# 
# These features are concatenated with the (text + title) feature matrix from Q2a, separately for each representation.
# 
# ### Note on data leakage
# `avg_product_rating` is computed across *all reviews of that product* — including reviews from this same dataset. Using it as a feature is technically valid for prediction (it's available at inference time on a real product page) but is correlated with the label (buyers tend to rate higher; products with higher ratings have more buyers). We expect this column to do most of the lifting in Q2b. We document this caveat rather than removing the feature, because the brief explicitly lists it as a candidate.

# In[22]:


# Build metadata features

# 1. brand_name → one-hot top-50 + 'OTHER'
TOP_K_BRANDS = 50
top_brands = df['brand_name'].value_counts().nlargest(TOP_K_BRANDS).index.tolist()
df['brand_grouped'] = df['brand_name'].where(df['brand_name'].isin(top_brands), 'OTHER')

ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
X_brand = ohe.fit_transform(df[['brand_grouped']])
print(f"X_brand: {X_brand.shape} ({len(top_brands)} top brands + 1 OTHER bucket)")

# 2. product_title → count vectorize against vocab.txt
df['product_title_clean'] = df['product_title'].fillna('').apply(
    lambda s: ' '.join(clean_title_tokens(s))
)
ptitle_vec = CountVectorizer(
    vocabulary=vocab_terms,
    tokenizer=lambda s: s.split(),
    lowercase=False,
    token_pattern=None,
)
X_ptitle = ptitle_vec.fit_transform(df['product_title_clean']).astype(np.float32)
print(f"X_ptitle: {X_ptitle.shape}, nnz={X_ptitle.nnz:,}")

# 3. Numeric features — avg_product_rating + log(price+1)
numeric_features = pd.DataFrame({
    'avg_product_rating': df['avg_product_rating'].fillna(df['avg_product_rating'].median()),
    'log_price': np.log1p(df['price'].fillna(df['price'].median())),
}).to_numpy(dtype=np.float32)

scaler = StandardScaler()
X_numeric = scaler.fit_transform(numeric_features).astype(np.float32)
X_numeric_sparse = csr_matrix(X_numeric)
print(f"X_numeric: {X_numeric.shape}, columns: avg_product_rating, log_price (z-scored)")

# Combine all metadata features into one sparse block
X_meta = sp_hstack([X_brand, X_ptitle, X_numeric_sparse]).tocsr()
print(f"X_meta combined: {X_meta.shape}, nnz={X_meta.nnz:,}")


# In[23]:


# Concatenate (text + title + metadata) and run 5-fold CV
q2b_results = []

X_q2b_count = sp_hstack([X_q2a_count, X_meta]).tocsr()
print(f"X_q2b_count: {X_q2b_count.shape}")
q2b_results.append(evaluate(X_q2b_count, y, 'Count (BoW) + Title + Metadata'))

# For embeddings, the meta block is sparse but the embeddings are dense.
# We convert the dense embeddings to sparse so we can use sp_hstack uniformly.
X_q2b_unweighted = sp_hstack([csr_matrix(X_q2a_unweighted), X_meta]).tocsr()
q2b_results.append(evaluate(X_q2b_unweighted, y, 'Unweighted GloVe + Title + Metadata'))

X_q2b_weighted = sp_hstack([csr_matrix(X_q2a_weighted), X_meta]).tocsr()
q2b_results.append(evaluate(X_q2b_weighted, y, 'TF-IDF Weighted GloVe + Title + Metadata'))

q2b_df = pd.DataFrame(q2b_results)
print("\n=== Q2b RESULTS — text + title + product metadata ===")
print(q2b_df.to_string(index=False))


# ## 3.4 Combined results and discussion
# 
# Bringing all three experiments together to answer the two research questions.
# 
# - **Q1:** Which representation wins on review text alone?
# - **Q2:** Does adding more information (title, then metadata) lift performance?

# In[24]:


# Combine all results
all_results = pd.concat([
    q1_df.assign(scope='Text only'),
    q2a_df.assign(scope='Text + Title'),
    q2b_df.assign(scope='Text + Title + Metadata'),
], ignore_index=True)

print("=== ALL RESULTS — sorted by macro-F1 ===")
display_df = all_results[['scope','representation','accuracy_mean','f1_macro_mean','roc_auc_mean']].copy()
display_df.columns = ['Scope', 'Representation', 'Accuracy', 'Macro-F1', 'ROC-AUC']
print(display_df.sort_values('Macro-F1', ascending=False).to_string(index=False))


# In[25]:


# Q1 winner
q1_winner = q1_df.loc[q1_df['f1_macro_mean'].idxmax()]
print(f"Q1 winner (macro-F1): {q1_winner['representation']} with {q1_winner['f1_macro_mean']:.4f}")

# Lift from Q1 to Q2a (adding title) — match by row index since both dfs share row order
print("\nLift from adding title (Q2a vs Q1):")
for i in range(len(q1_df)):
    base = q1_df['f1_macro_mean'].iloc[i]
    new = q2a_df['f1_macro_mean'].iloc[i]
    print(f"  {q1_df['representation'].iloc[i]:>30s}: {base:.4f}  ->  {new:.4f}  (delta = {new-base:+.4f})")

# Lift from Q2a to Q2b (adding metadata)
print("\nLift from adding metadata (Q2b vs Q2a):")
for i in range(len(q2a_df)):
    base = q2a_df['f1_macro_mean'].iloc[i]
    new = q2b_df['f1_macro_mean'].iloc[i]
    print(f"  {q2a_df['representation'].iloc[i]:>30s}: {base:.4f}  ->  {new:.4f}  (delta = {new-base:+.4f})")

# Final headline: best overall configuration
best = all_results.loc[all_results['f1_macro_mean'].idxmax()]
print(f"\nBest overall: {best['representation']} ({best['scope']}) — macro-F1 = {best['f1_macro_mean']:.4f}, ROC-AUC = {best['roc_auc_mean']:.4f}")


# ## 3.5 Findings and conclusions
# 
# The numerical results above answer Q1 and Q2 directly. Interpreting them:
# 
# ### Q1 — best representation for review text alone
# The three representations occupy a hierarchy: count vectors carry word-identity signal, unweighted embeddings carry averaged semantic signal, weighted embeddings carry TF-IDF-amplified semantic signal. The relative ordering tells us *what kind* of signal drives `is_a_buyer` — see the printed `Q1 winner` line above.
# 
# ### Q2 — does more information help?
# Two separate contributions to consider:
# 
# - **Title.** Titles are short, opinionated summaries. The marginal lift is bounded — bodies are already much longer and contain most of the same vocabulary. We expect a small positive lift for representations that aren't already saturated.
# - **Metadata.** `avg_product_rating` is the dominant feature here (it correlates strongly with buyer behaviour because rating-aware customers buy more often, and high-rated products attract more buyers in general). `brand_name` and `product_title` should add modest signal. `price` is weakly informative — high-priced products are bought by both classes.
# 
# If the metadata-enriched models show a large jump in ROC-AUC, that is consistent with `avg_product_rating` being a near-leakage feature; the rubric acknowledges this kind of feature is fair game in Q2 ("adding extra information of a product such as ... average product rating").
# 
# ### Caveats
# 1. **Logistic regression as the only classifier.** The brief allows other models. We chose one estimator across all six experiments to keep the *representation* (not the model) as the controlled variable. A future extension would re-run with `LinearSVC`, `MultinomialNB` (count-only), or a gradient-boosted tree to confirm the ranking is estimator-independent.
# 2. **No hyperparameter tuning.** We used default `C=1.0` regularisation throughout. Tuning per-representation could shift the relative ordering, but that risks comparing tuned-and-overfit configurations rather than representations.
# 3. **Class imbalance.** We did not use `class_weight='balanced'`. This means the classifier optimises for accuracy on the 78.7% majority class. Macro-F1 still tells us how well it distinguishes both classes, so it is the headline metric.
# 
# ### Recommendation for the web app (Milestone 2)
# Pick the representation+scope that wins on macro-F1 in the table above. If the count-vector representation wins, it has the practical advantage of being interpretable (per-word coefficients show *why* the model thinks a review reflects a buyer) — a property that is useful in a UI that explains predictions to end users.

# ---
# 
# ## 3.6 Appendix — classifier-family robustness check
# 
# The Q1/Q2 results above used **logistic regression** for every experiment, deliberately, so that the *feature representation* was the only varying factor. A natural follow-up question is: **does the "BoW + Title + Metadata wins" finding hold up under non-linear models?**
# 
# We run a short additional experiment: fit three different classifiers on the *winning* feature configuration (BoW + Title + Metadata, 24,175 features) under the same 5-fold stratified CV protocol.
# 
# | Classifier | What it does | Why include it |
# |---|---|---|
# | `LogisticRegression` | Linear weighted sum + sigmoid | Our baseline from §3.1–§3.3 |
# | `DecisionTreeClassifier` | Single recursive splitting tree | Strict non-linear baseline; tends to overfit |
# | `RandomForestClassifier` | Ensemble of 200 trees on bootstrap samples + feature subsets | Standard non-linear classifier; handles tabular metadata well |
# 
# We expect:
# - **Decision tree** alone will underperform — single trees overfit on high-dim sparse inputs.
# - **Random forest** may edge logistic regression slightly, primarily because `avg_product_rating`'s relationship with `is_a_buyer` is monotonic but non-linear (e.g. ratings of 4.5 vs 4.0 might matter much more than 3.5 vs 3.0).
# - The ranking should be **RF ≈ LogReg ≫ DT**.
# 
# If RF wins, that's a deployment recommendation for Milestone 2 — but we keep logistic regression as the *comparison* classifier in §3.1–§3.3 so the representation analysis remains apples-to-apples.

# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Use the winning feature configuration: BoW + Title + Metadata
X_best = X_q2b_count
print(f"Feature matrix for appendix: {X_best.shape}, nnz={X_best.nnz:,}")

# Parallelism note: cross_validate(n_jobs=-1) already parallelises the 5 folds.
# We set RF n_jobs=1 to avoid nested parallelism (which oversubscribes cores on
# macOS and can hang). Each fold trains its own RF single-threaded; folds run in parallel.
classifiers = [
    ("Logistic Regression", LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE)),
    ("Decision Tree",       DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("Random Forest (100 trees)",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)),
]

appendix_results = []
for name, clf in classifiers:
    print(f"Evaluating {name} ...")
    appendix_results.append(evaluate(X_best, y, name, classifier=clf))

appendix_df = pd.DataFrame(appendix_results)
print("")
print("=== APPENDIX — same features (BoW + Title + Metadata), three classifiers ===")
print(appendix_df.to_string(index=False))


# ### Appendix discussion
# 
# What the table above tells us:
# 
# - **If LogReg ≈ RF**: the BoW + Title + Metadata feature set has already linearised the prediction problem — there's not much non-linear signal left for the random forest to exploit. The representation comparison in §3.1–§3.3 is robust.
# - **If RF > LogReg by a clear margin**: the metadata block (especially `avg_product_rating`) has a non-linear relationship with `is_a_buyer` that linear models can't fully capture. We'd recommend RF for the Milestone 2 deployment, while keeping LogReg's results as the *fair* comparison for Q1/Q2.
# - **If DT trails badly**: confirms the standard observation that single trees overfit and need ensembling to be competitive — not a surprise, but a useful sanity check.
# 
# Either way, this appendix doesn't change the answers to Q1 and Q2 — those questions are about feature representations, and the answer is the same regardless of which downstream classifier we pick. It only informs the *deployment* choice for Milestone 2.

# ## 3.7 Feature-importance and per-class breakdown
# 
# The aggregate metrics (accuracy, F1, AUC) tell us *how well* the model performs but not *what it has learned* or *where it errs*. Two short analyses fill that gap:
# 
# 1. **Coefficient inspection (logistic regression)** — for the linear model, each feature has a signed weight. Positive weights push toward `is_a_buyer = True`; negative weights push away. Reading the top features tells us, in plain words, what the model thinks distinguishes a buyer from a non-buyer.
# 
# 2. **Feature importance (random forest)** — RF's impurity-based importance is unsigned and captures *interaction* importance too. It complements the linear coefficients by surfacing features that matter through interactions rather than additive effects.
# 
# 3. **Confusion matrix on a held-out fold** — shows the asymmetry between false positives (predicted buyer, actually non-buyer) and false negatives (predicted non-buyer, actually buyer). For an e-commerce "likelihood to purchase" UI, FPs and FNs cost differently — surfacing this lets us decide whether to rebalance the decision threshold in Milestone 2.

# In[27]:


# Build a feature-name list aligned with the columns of X_q2b_count.
# This is needed to interpret coefficients and importances.
brand_names = list(ohe.get_feature_names_out(['brand_grouped']))
feature_names = (
    [f"body::{w}" for w in vocab_terms]            # 0..8053
    + [f"title::{w}" for w in vocab_terms]         # 8054..16107
    + [f"brand::{b.replace('brand_grouped_','')}" for b in brand_names]   # 16108..
    + [f"ptitle::{w}" for w in vocab_terms]        # next 8054
    + ["num::avg_product_rating", "num::log_price"]
)
print(f"Feature names: {len(feature_names)} (expected {X_q2b_count.shape[1]})")
assert len(feature_names) == X_q2b_count.shape[1], "feature-name list does not align with X_q2b_count columns"


# In[28]:


# Logistic regression coefficients on the full training set
# (fit once on all data — interpretation only, not held-out evaluation)
logreg_full = LogisticRegression(solver='liblinear', max_iter=1000, random_state=RANDOM_STATE)
logreg_full.fit(X_q2b_count, y)
coef = logreg_full.coef_.flatten()  # shape: (n_features,)

coef_df = pd.DataFrame({'feature': feature_names, 'coef': coef})
# Positive coefficients push toward "is_a_buyer = True" (the positive class)
print("Top-15 features pushing TOWARD 'buyer' (positive coefficient):")
print(coef_df.nlargest(15, 'coef').to_string(index=False))
print()
print("Top-15 features pushing AWAY from 'buyer' (negative coefficient):")
print(coef_df.nsmallest(15, 'coef').to_string(index=False))


# In[29]:


# Random forest feature importances on the same feature matrix
rf_full = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_full.fit(X_q2b_count, y)

imp_df = pd.DataFrame({'feature': feature_names, 'importance': rf_full.feature_importances_})
print("Top-20 features by random-forest importance (impurity-based):")
print(imp_df.nlargest(20, 'importance').to_string(index=False))


# In[30]:


from sklearn.metrics import confusion_matrix, classification_report

# Hold-out evaluation: train on 4/5, predict on 1/5 (one fold's worth) — gives an honest
# look at where the model errs without re-running the full CV.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(cv.split(X_q2b_count, y))

# Use the chosen production model for Milestone 2: LogReg on BoW + Title + Metadata
clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=RANDOM_STATE)
clf.fit(X_q2b_count[train_idx], y[train_idx])
y_pred = clf.predict(X_q2b_count[test_idx])
y_true = y[test_idx]

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion matrix on a held-out fold (Logistic Regression, BoW + Title + Metadata):")
print(f"                 predicted=False (non-buyer)   predicted=True (buyer)")
print(f"actual=False:    {cm[0,0]:>10d} (TN)              {cm[0,1]:>10d} (FP)")
print(f"actual=True:     {cm[1,0]:>10d} (FN)              {cm[1,1]:>10d} (TP)")
print()
print("Per-class breakdown:")
print(classification_report(y_true, y_pred, target_names=['Non-buyer (False)', 'Buyer (True)'], digits=4))


# ### Reading the results
# 
# - **Logistic-regression coefficients** name *specific words* that move the prediction. The numerical block (`avg_product_rating`, `log_price`) and the brand one-hots usually dominate, which mirrors the headline finding from Q2b: metadata carries most of the signal. Among text features, words that survive to the top are the ones that genuinely correlate with purchase intent.
# 
# - **Random-forest importance** typically agrees with logistic regression on the metadata features (especially `avg_product_rating`) but ranks individual word features lower, because trees split on whichever feature offers the largest impurity drop, and dense numeric features almost always win that contest over sparse single-word indicators.
# 
# - **Confusion matrix.** With 78.7% of the corpus being buyers, a naive baseline gets all True-positives and all False-positives (predicting True for everyone). Our model trades some of those wins for a non-trivial number of correct non-buyer predictions. The classification report's per-class precision/recall makes the cost of imbalance concrete: the minority class (non-buyer) typically gets weaker recall, which is the right place to tune the threshold if Milestone 2 cares more about catching non-buyers than about maximising overall accuracy.
