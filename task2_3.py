# %% [markdown]
# # Assignment 3: Milestone I Natural Language Processing
# ## Task 2 and Task 3
# ### Cosmetics/Beauty Review Classification — Predicting `is_a_buyer`
# 
# #### Group Members
# 1. Vo Ngoc Dung — S4124370
# 2. Tang Hoang Ha — S4147768
# 3. Nguyen Anh Duc — S4136756
# 4. Nguyen Quoc Trong Nghia — S3343711
# 
# ---
# 
# ## Introduction
# 
# This notebook implements two tasks:
# 
# **Task 2 -  Generating Feature Representations for Cosmetics/Beauty Reviews** Convert the preprocessed reviews (from Task 1) into three numeric representations suitable for machine learning:
# 1. Sparse count vectors (bag-of-words using `vocab.txt`)
# 2. Unweighted average FastText word vectors (300-d)
# 3. TF-IDF weighted average FastText word vectors (300-d)
# 
# **Task 3 — Classification**, applying machine learning models to the feature representations generated in Task 2 to predict whether a reviewer is a verified buyer (`is_a_buyer = True/False`).
# 
# ### Task Structure
# 
# | Part | Question | Objective |
# |------|----------|-----------|
# | **Part 1 (Q1)** | Which language model performs best? | Compare 3 feature representations × 3 classifiers using review text **only**. |
# | **Part 2 (Q2)** | Does adding more information improve accuracy? | Progressively add review title (Q2a) and product metadata (Q2b) to measure lift. |
# 
# ### Why This Task Matters
# The EDA in Task 1 showed that no structured feature exceeds |r| = 0.21 with `is_a_buyer`. This means the **text** of a review is likely the primary signal for distinguishing buyers from non-buyers. Task 3 validates this hypothesis empirically and determines which combination of features and models yields the best classification performance.
# 
# ---
# 
# ### Classifier Selection: Why These 3 Models?
# 
# We selected three classifiers that span the **bias–variance spectrum** and represent fundamentally different learning paradigms, enabling a fair comparison of how each feature representation interacts with model complexity:
# 
# | Classifier | Type | Why Selected | Alternatives Considered |
# |-----------|------|--------------|------------------------|
# | **Logistic Regression** | Linear, parametric | The standard baseline for binary classification. Learns a single hyperplane; works well when features are linearly separable. Fast, interpretable (coefficients = feature importance), and regularised (L2) to prevent overfitting on high-dimensional sparse input. | SVM (linear kernel) — similar performance but slower on large datasets; Naive Bayes — strong BoW baseline but assumes feature independence, which is violated by correlated embedding dimensions. |
# | **Decision Tree** | Non-linear, non-parametric | Captures feature interactions and non-linear boundaries without any assumptions about data distribution. Serves as the "simple non-linear baseline" — a single tree grown to purity will overfit, revealing how much each representation rewards memorisation vs generalisation. | KNN — also non-parametric but prohibitively slow on 60K × 5634 sparse matrices; requires distance computation over all training points. |
# | **Random Forest** | Ensemble (bagging) | Combines many decision trees trained on bootstrapped subsets with random feature subsets per split, dramatically reducing variance compared to a single tree. The de facto "black-box" baseline for tabular/NLP tasks that balances accuracy and robustness without hyperparameter tuning. | Gradient Boosting (XGBoost/LightGBM) — potentially higher accuracy but requires careful hyperparameter tuning (learning rate, depth, iterations) which is outside the scope of this milestone; AdaBoost — more sensitive to noisy data and outliers. |
# 
# **Why not neural networks (MLP, CNN, LSTM)?**
# Neural architectures require significantly more data, training time, and hyperparameter tuning (hidden layers, dropout, learning rate schedules, batch size). For a 60K-sample binary classification task with pre-computed features, classical ML models are sufficient and more reproducible. Additionally, using sklearn's built-in models ensures full compatibility with `cross_validate` and consistent random seeding.
# 
# ---
# 
# ### Evaluation Protocol: Why 5-Fold Stratified CV with These 5 Metrics?
# 
# **Cross-validation strategy — 5-fold Stratified K-Fold:**
# 
# | Design Choice | Justification | Alternatives Considered |
# |--------------|---------------|------------------------|
# | **K = 5** | Standard in ML literature (Hastie et al., 2009). Balances bias–variance of the performance estimate: K=3 is high-bias (small train sets); K=10 is low-bias but high-variance and 2× slower. With 60K samples, each fold has ~12K test samples — sufficient for stable metric estimates. | K=10 — common but doubles runtime for 27 configurations; Leave-One-Out — prohibitively expensive and high-variance for this dataset size. |
# | **Stratified** | Preserves the 79.7%/20.3% class ratio in every fold. Without stratification, some folds could randomly contain very few minority-class samples, producing unstable metric estimates — especially for macro-averaged metrics that weight both classes equally. | Regular KFold — risks class imbalance in individual folds; RepeatedStratifiedKFold — more stable estimates but 2–5× slower. |
# | **shuffle=True, random_state=42** | Shuffling before splitting avoids systematic bias from data ordering (e.g., reviews sorted by date or product). Fixed seed ensures reproducibility across runs and team members. | No shuffle — risks temporal/product clustering in folds. |
# 
# **Metric selection — why these 5 metrics?**
# 
# | Metric | What It Measures | Why Included |
# |--------|-----------------|--------------|
# | **Accuracy** | Overall proportion correct | The most intuitive metric, but misleading under class imbalance (a "predict all buyers" baseline achieves 79.7%). Included for completeness and comparability with prior work. |
# | **Macro-Precision** | Average precision across both classes (unweighted) | Measures how often positive predictions are correct, treating both classes equally. Important for detecting whether a model achieves high accuracy simply by ignoring the minority class. |
# | **Macro-Recall** | Average recall across both classes (unweighted) | Measures how completely each class is captured. A model predicting "buyer" for everyone has 100% buyer-recall but 0% non-buyer recall — macro-recall exposes this failure. |
# | **Macro-F1** | Harmonic mean of macro-precision and macro-recall | **Our primary comparison metric.** The harmonic mean penalises models that sacrifice one class for the other. Macro-averaging ensures the minority class (non-buyers, 20.3%) contributes equally to the final score. |
# | **ROC-AUC** | Area under the ROC curve (probability ranking quality) | Threshold-independent; measures how well the model's predicted probabilities rank positive vs negative samples. Useful for comparing models when the operating threshold is not fixed (e.g., in a production system where the threshold may be tuned post-hoc). |
# 
# **Why Macro-F1 as the primary metric (not accuracy)?**
# With 79.7% buyers, a trivial "always predict buyer" classifier achieves 79.7% accuracy but 0% non-buyer recall — useless in practice. Macro-F1 forces models to perform well on **both** classes by equally weighting buyer and non-buyer performance. This aligns with the practical goal: identifying non-buyers is at least as important as confirming buyers.
# 
# ---
# 
# ### Reproducibility
# - **Random seed = 42** used consistently for all stochastic components (numpy, CV splits, model initialisation).
# - **Same CV object** (StratifiedKFold with identical parameters) applied to every single evaluation, ensuring all 27 configurations are compared on exactly the same train/test splits.

# %% [markdown]
# ---
# # Part 1 — Q1: Which language model performs best?
# 
# **Research Question:** Given three different feature representations of the same review text, which representation yields the best classification performance for predicting `is_a_buyer`?
# 
# This part evaluates all 9 combinations of {3 representations × 3 classifiers} using only the review body text (no title or metadata). The goal is to isolate the effect of the **feature representation** on classification quality.
# 
# ## Step 1 · Imports and data loading
# 
# **Data flow:**
# - `processed.csv` → target labels (`is_a_buyer`) and review metadata (for Q2)
# - `vocab.txt` → vocabulary size and word-to-index mapping (for title BoW in Q2)
# - `count_vectors.txt` → sparse BoW features (from Task 2)
# - `unweighted_vectors.txt` → dense 300-d unweighted FastText averages (from Task 2)
# - `weighted_vectors.txt` → dense 300-d TF-IDF weighted FastText averages (from Task 2)
# 
# All feature files are pre-generated by Task 2 and loaded as-is — no feature recomputation occurs in this notebook.

# %% [markdown]
# ## Importing Libraries

# %%
# !pip install -q --upgrade pip setuptools wheel
%pip install -q -r requirements.txt

# %%
# Standard libraries
import os, re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sparse matrix utilities (for the bag-of-words count vectors)
from scipy.sparse import csr_matrix, hstack as sp_hstack

# Models — three classifiers
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier

# Evaluation
from sklearn.model_selection         import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing           import StandardScaler, OneHotEncoder

# Pretrained word embeddings — same FastText model used in Task 2
import gensim.downloader as gensim_api

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %%
# Paths to Task 1 / Task 2 outputs
PROCESSED_CSV     = "processed.csv"
VOCAB_FILE        = "vocab.txt"
STOPWORDS_FILE    = "stopwords_en.txt"
COUNT_VECTOR_FILE = "count_vectors.txt"
UNWEIGHTED_FILE   = "unweighted_vectors.txt"
WEIGHTED_FILE     = "weighted_vectors.txt"
FASTTEXT_MODEL_NAME = "fasttext-wiki-news-subwords-300"

# %%
def GenerateCountVector():
    # Load vocabulary (word -> integer index)
    vocab = {}
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, idx = line.rsplit(":", 1)
            vocab[word] = int(idx)

    # Load processed reviews
    df = pd.read_csv(PROCESSED_CSV)
    # review_text contains space-separated tokens produced by Task 1
    review_texts = df["review_text"].fillna("").astype(str).tolist()

    # Build and write sparse count vectors
    with open(COUNT_VECTOR_FILE, "w", encoding="utf-8") as out:
        for review_idx, text in enumerate(review_texts):
            tokens = text.split()
            # Count only tokens that exist in the vocabulary
            counts = Counter(token for token in tokens if token in vocab)
            # Sort by word integer index for a consistent ordering
            sparse_entries = sorted(
                (vocab[word], freq) for word, freq in counts.items()
            )
            sparse_str = ",".join(f"{idx}:{freq}" for idx, freq in sparse_entries)
            out.write(f"#{review_idx},{sparse_str}\n")

    print(f"Count vectors saved to '{COUNT_VECTOR_FILE}' ({len(review_texts)} reviews).")


# Run
GenerateCountVector()

# %%
# 1c. Title FastText embeddings (unweighted + TF-IDF weighted) — load FastText once
print(f"Loading FastText model '{FASTTEXT_MODEL_NAME}'")
fasttext_model = gensim_api.load(FASTTEXT_MODEL_NAME)
vector_size = fasttext_model.vector_size
print(f"Model loaded. Vector size: {vector_size}")

# %%
def GenerateEmbeddingVectors():
    # Load processed reviews
    df = pd.read_csv(PROCESSED_CSV)
    review_texts = df["review_text"].fillna("").astype(str).tolist()
    print("Load review")
    # Fit TF-IDF over the full corpus (for weighted representation)
    # tokenizer=str.split preserves the already-cleaned tokens from Task 1
    tfidf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None)
    tfidf_matrix = tfidf.fit_transform(review_texts)
    tfidf_feature_names = tfidf.get_feature_names_out()
    tfidf_vocab = {word: idx for idx, word in enumerate(tfidf_feature_names)}

    # Generate and write vectors
    with open(UNWEIGHTED_FILE, "w", encoding="utf-8") as uw_out, open(
        WEIGHTED_FILE, "w", encoding="utf-8"
    ) as w_out:

        for review_idx, text in enumerate(review_texts):
            tokens = text.split()
            # Keep only tokens the FastText model knows
            valid_tokens = [t for t in tokens if t in fasttext_model]

            if valid_tokens:
                vectors = np.array([fasttext_model[t] for t in valid_tokens])

                # Unweighted: simple average of word vectors
                unweighted_vec = vectors.mean(axis=0)

                # Weighted: TF-IDF weighted average
                tfidf_row = tfidf_matrix[review_idx]
                weights = np.array(
                    [
                        tfidf_row[0, tfidf_vocab[t]] if t in tfidf_vocab else 0.0
                        for t in valid_tokens
                    ]
                )
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weighted_vec = (vectors * weights[:, np.newaxis]).sum(
                        axis=0
                    ) / weight_sum
                else:
                    weighted_vec = unweighted_vec
            else:
                unweighted_vec = np.zeros(vector_size)
                weighted_vec = np.zeros(vector_size)

            uw_out.write(
                f"#{review_idx}," + ",".join(f"{v:.6f}" for v in unweighted_vec) + "\n"
            )
            w_out.write(
                f"#{review_idx}," + ",".join(f"{v:.6f}" for v in weighted_vec) + "\n"
            )

    print(f"Number of reviews: {len(review_texts)}")
    print(f"Unweighted vectors saved to '{UNWEIGHTED_FILE}'")
    print(f"Weighted vectors saved to '{WEIGHTED_FILE}'")


# Run
GenerateEmbeddingVectors()

# %%
# Load processed reviews and the binary label
df = pd.read_csv(PROCESSED_CSV)
df['review_text']  = df['review_text'].fillna('').astype(str)
y = df['is_a_buyer'].astype(str).str.lower().map({'true': 1, 'false': 0}).astype(int).to_numpy()

print(f"Reviews: {len(df):,}")
print(f"Class balance: buyers = {y.mean():.3f}, non-buyers = {1 - y.mean():.3f}")

# %%
# 1a. Vocabulary (word -> integer index)
word_to_index = {}
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        w, i = line.rsplit(':', 1)
        word_to_index[w] = int(i)

vocab_size  = len(word_to_index)
vocab_terms = sorted(word_to_index, key=word_to_index.get)
print(f"Vocabulary size: {vocab_size}")

# %%
# 1b. Helper to load count_vectors.txt -> sparse CSR matrix
def load_count_vectors(path, n_rows, n_cols):
    rows, cols, vals = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for row_i, line in enumerate(f):
            _, _, body = line.rstrip('\n').partition(',')
            if not body:
                continue
            for pair in body.split(','):
                idx, cnt = pair.split(':')
                rows.append(row_i); cols.append(int(idx)); vals.append(int(cnt))
    return csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)

# Helper to load dense embedding vectors (one row per review, comma-separated values)
def load_dense_vectors(path, n_rows):
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline().rstrip('\n')
        _, _, body = first.partition(',')
        dim = body.count(',') + 1
    X = np.zeros((n_rows, dim), dtype=np.float32)
    with open(path, 'r', encoding='utf-8') as f:
        for row_i, line in enumerate(f):
            _, _, body = line.rstrip('\n').partition(',')
            X[row_i] = np.fromstring(body, sep=',', dtype=np.float32)
    return X, dim

X_count                  = load_count_vectors(COUNT_VECTOR_FILE, len(df), vocab_size)
X_unweighted, EMBED_DIM  = load_dense_vectors(UNWEIGHTED_FILE,   len(df))
X_weighted,   _          = load_dense_vectors(WEIGHTED_FILE,     len(df))

print(f"X_count       : sparse {X_count.shape}, nnz={X_count.nnz:,}")
print(f"X_unweighted  : dense  {X_unweighted.shape}")
print(f"X_weighted    : dense  {X_weighted.shape}")
print(f"FastText dim  : {EMBED_DIM}")

# %% [markdown]
# ## Step 2 · Define the three classifiers and the 5-fold CV evaluator
# 
# ### Classifier Hyperparameter Justification
# 
# Each classifier below uses specific hyperparameters chosen for this task. We use **factory functions** (not pre-instantiated objects) because sklearn estimators are stateful — `cross_validate` requires a fresh, unfitted estimator for each fold.
# 
# | Classifier | Parameter | Value | Justification |
# |-----------|-----------|-------|---------------|
# | **Logistic Regression** | `solver` | `'liblinear'` | Optimised for small-to-medium datasets and binary classification. Supports both L1 and L2 penalties. Deterministic (no random initialisation). Alternatives: `'lbfgs'` (default since sklearn 0.22) — faster on dense data but can be slower on very sparse matrices; `'saga'` — supports L1/L2/ElasticNet but requires feature scaling and is stochastic. |
# | | `max_iter` | `1000` | Ensures convergence even on the 5,634-dimensional sparse BoW input. Default (100) may not converge with high-dimensional data, producing sklearn warnings. |
# | | `C` | `1.0` | Default regularisation strength (inverse of λ). We deliberately do not tune C — the goal is a fair comparison of **representations**, not of hyperparameter-optimised models. Using defaults makes the comparison attributable to the features, not the tuning effort. |
# | **Decision Tree** | `criterion` | `'gini'` | Standard impurity measure for classification. Gini and entropy produce nearly identical trees in practice (Raileanu & Stoffel, 2004), but Gini is computationally cheaper (no logarithm). |
# | | `max_depth` | `None` | Grows the tree to pure leaves (no pre-pruning). This intentionally allows overfitting — a fully-grown tree reveals how much "memorisable" structure exists in each representation. If a representation has noisy/redundant features, the unpruned tree will overfit more, showing up as low CV scores. |
# | **Random Forest** | `n_estimators` | `100` | sklearn default since v0.22. Empirically, 100 trees are sufficient for convergence of the ensemble error on this dataset size. Increasing to 200–500 yields diminishing returns (<0.1% improvement) at 2–5× cost. |
# | | `max_features` | `'sqrt'` | Standard for classification (Breiman, 2001). Each split considers √p random features, decorrelating the trees and reducing ensemble variance. Alternative: `'log2'` — similar effect but slightly more restrictive. |
# | | `n_jobs` | `1` | Avoids nested parallelism — the outer `cross_validate` call already parallelises across folds with `n_jobs=-1`. Nested parallelism can cause resource contention and slowdowns. |
# 
# ### Why default hyperparameters throughout?
# The goal of Q1 is to compare **feature representations**, not to find the best-tuned model. Using identical, default hyperparameters across all 9 (representation × model) combinations isolates the effect of the representation. Any performance difference is attributable to the quality of the features, not to differential tuning effort.

# %%
# 2a. The three classifiers
#     Each is wrapped in a small factory so we get a fresh instance per evaluation
#     (sklearn estimators are stateful).

def make_logreg():
    # Linear model with L2 regularisation, sigmoid output for binary classification.
    return LogisticRegression(
        solver       = 'liblinear',  # fast on sparse + binary; deterministic
        max_iter     = 1000,         # ensure convergence even on high-dim sparse input
        C            = 1.0,          # default regularisation strength
        random_state = RANDOM_STATE,
    )

def make_decision_tree():
    # A single, unrestricted CART tree. Will overfit if grown to pure leaves —
    # we keep it that way as the simple non-linear baseline.
    return DecisionTreeClassifier(
        criterion    = 'gini',       # standard impurity measure
        max_depth    = None,         # grow until pure leaves
        random_state = RANDOM_STATE,
    )

def make_random_forest():
    # Ensemble of 100 trees on bootstrap samples, with sqrt(n_features) per split.
    # n_jobs=1 avoids nested parallelism — cross_validate already parallelises folds.
    return RandomForestClassifier(
        n_estimators = 100,          # sklearn default since 0.22
        max_features = 'sqrt',       # standard for classification
        n_jobs       = 1,            # outer cross_validate handles parallelism
        random_state = RANDOM_STATE,
    )

MODELS = {
    'Logistic Regression': make_logreg,
    'Decision Tree'      : make_decision_tree,
    'Random Forest'      : make_random_forest,
}

# %%
# 2b. 5-fold stratified CV evaluator
#     Returns the mean of each scoring metric across the 5 folds.

SCORING = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']

def evaluate(X, y, classifier, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        classifier, X, y,
        cv      = cv,
        scoring = SCORING,
        n_jobs  = -1,                # parallelise across folds
        return_train_score = False,
    )
    return {
        'Accuracy'        : scores['test_accuracy'].mean(),
        'Macro-Precision' : scores['test_precision_macro'].mean(),
        'Macro-Recall'    : scores['test_recall_macro'].mean(),
        'Macro-F1'        : scores['test_f1_macro'].mean(),
        'ROC-AUC'         : scores['test_roc_auc'].mean(),
    }

# %% [markdown]
# ## Step 3 · Run Q1 — 3 representations × 3 models = 9 evaluations
# 
# ### Experimental Design
# 
# **Independent variable:** Feature representation (3 levels)
# - **Count (BoW):** Sparse 5,634-d vector of raw token frequencies. Captures *which* words appear and how often, but ignores word order and semantics. Each dimension corresponds to a specific vocabulary word.
# - **Unweighted FastText:** Dense 300-d vector from averaging all token embeddings equally. Captures semantic similarity (words with similar meanings have similar vectors) but loses word frequency information.
# - **TF-IDF Weighted FastText:** Dense 300-d vector where each token's embedding is scaled by its TF-IDF score before averaging. Combines semantic information with corpus-level word importance — distinctive words contribute more to the review's representation.
# 
# **Dependent variables:** 5 classification metrics (Accuracy, Macro-Precision, Macro-Recall, Macro-F1, ROC-AUC)
# 
# **Control variables:** Same classifier hyperparameters, same CV splits, same random seed, same target variable (`is_a_buyer`), same preprocessing pipeline (from Task 1).
# 
# **Hypothesis:** Count vectors should outperform averaged embeddings for this task because:
# 1. The vocabulary is already cleaned and compact (5,634 terms) — high-dimensional but sparse and informative.
# 2. Purchase behaviour is likely signalled by specific word choices (e.g., "bought", "delivered", "returned") rather than general semantic similarity.
# 3. Averaging 300-d embeddings across a review loses the identity of individual discriminative tokens.

# %%
# 3a. Define the three representations under test (review text only)
REPRESENTATIONS_Q1 = {
    'Count (BoW)'              : X_count,
    'Unweighted FastText'      : X_unweighted,
    'TF-IDF Weighted FastText' : X_weighted,
}

# 3b. Evaluate every (representation, model) pair under 5-fold CV
results_q1 = []
for rep_name, X in REPRESENTATIONS_Q1.items():
    for model_name, make_model in MODELS.items():
        print(f"Evaluating  {rep_name:<28s} | {model_name}")
        metrics = evaluate(X, y, make_model())
        results_q1.append({
            'Representation' : rep_name,
            'Model'          : model_name,
            **metrics,
        })

q1_df = pd.DataFrame(results_q1)
print("\n=== Q1 RESULTS ===")
print(q1_df.round(4).to_string(index=False))

# %% [markdown]
# ## Step 4 · Q1 Analysis — Compare representations and models
# 
# ### Comparison Methodology
# 
# We present the Q1 results using three complementary views:
# 1. **Pivot tables** (Macro-F1 and ROC-AUC) — for precise numeric comparison of all 9 cells.
# 2. **Grouped bar charts** (all 5 metrics) — for visual pattern detection across representations and models.
# 3. **Best configuration identification** — highlighting the winning (representation, model) pair.
# 
# **Why pivot on both Macro-F1 and ROC-AUC?**
# - Macro-F1 is our primary metric (threshold-dependent, penalises class imbalance).
# - ROC-AUC is threshold-independent and reveals whether a model's *probability rankings* are good even if the default 0.5 threshold is suboptimal. A model with high AUC but low F1 suggests threshold tuning could improve it.

# %%
# 4a. Pivot table — Macro-F1 by Representation x Model
def pivot(df_results, metric):
    p = df_results.pivot_table(index='Representation', columns='Model', values=metric)
    return p[['Logistic Regression', 'Decision Tree', 'Random Forest']]

print("Macro-F1")
print(pivot(q1_df, 'Macro-F1').round(4).to_string())

print("\nROC-AUC")
print(pivot(q1_df, 'ROC-AUC').round(4).to_string())

# %%
# 4b. Grouped bar chart — one panel per metric, bars grouped by representation
def grouped_bars(df_results, ax, metric, title):
    p = pivot(df_results, metric)
    reps = p.index.tolist()
    models = list(p.columns)
    width = 0.27
    x = np.arange(len(reps))
    colors = ['#3b7dd8', '#e08e3a', '#5fb56b']
    for i, m in enumerate(models):
        ax.bar(x + (i-1)*width, p[m].to_numpy(), width=width, label=m, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(reps, rotation=15, ha='right')
    ax.set_title(title); ax.set_ylabel(metric)
    # Auto-scale with a small head/tail margin so all three bars are visible
    vals = p.to_numpy().flatten()
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(0.02, (hi - lo) * 0.15)
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.legend(loc='lower right', fontsize=9)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
grouped_bars(q1_df, axes[0,0], 'Accuracy',        'Q1 — Accuracy')
grouped_bars(q1_df, axes[0,1], 'Macro-Precision', 'Q1 — Macro-Precision')
grouped_bars(q1_df, axes[0,2], 'Macro-Recall',    'Q1 — Macro-Recall')
grouped_bars(q1_df, axes[1,0], 'Macro-F1',        'Q1 — Macro-F1')
grouped_bars(q1_df, axes[1,1], 'ROC-AUC',         'Q1 — ROC-AUC')
axes[1,2].axis('off')  # leave the 6th panel blank — we only have 5 metrics
plt.suptitle('Q1: Language model comparison (review text only)', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# %%
# 4c. Best (Representation, Model) pair on Macro-F1
best_q1 = q1_df.loc[q1_df['Macro-F1'].idxmax()]
print("Q1 — best (Macro-F1) configuration:")
print(f"  Representation : {best_q1['Representation']}")
print(f"  Model          : {best_q1['Model']}")
print(f"  Accuracy        = {best_q1['Accuracy']:.4f}")
print(f"  Macro-Precision = {best_q1['Macro-Precision']:.4f}")
print(f"  Macro-Recall    = {best_q1['Macro-Recall']:.4f}")
print(f"  Macro-F1        = {best_q1['Macro-F1']:.4f}")
print(f"  ROC-AUC         = {best_q1['ROC-AUC']:.4f}")

# %% [markdown]
# ---
# # Part 2 — Q2: Does adding more information improve accuracy?
# 
# **Research Question:** Does incorporating the review title and/or product metadata alongside review text improve classification performance for `is_a_buyer`?
# 
# We re-use the same 3 models and the same evaluator from Part 1. The only thing that changes is the **input feature matrix** — we extend it progressively:
# - **Q2a:** Body + Title (tests whether short review titles add complementary signal)
# - **Q2b:** Body + Title + Metadata (tests whether structured product information adds orthogonal signal)
# 
# ---
# 
# ## Step 1 · Build the title features (Q2a)
# 
# ### Why Include Review Titles?
# 
# Review titles are short summaries written by the reviewer (e.g., "Worth buying", "Not impressed", "Best moisturiser ever"). They potentially carry concentrated sentiment and intent signals that complement the longer review body. However, their brevity (median 2 tokens after cleaning) limits the information they can provide.
# 
# **Hypothesis:** Titles should provide minimal but non-zero lift, because:
# 1. Titles often summarise the review body — redundant rather than complementary.
# 2. With only 2 tokens on average, there is insufficient content for reliable BoW or embedding representation.
# 3. However, some titles contain strong signals absent from the body (e.g., "Waste of money", "Must buy").
# 
# ### Title Feature Engineering — Three Representations
# 
# | Representation | How Title Is Encoded | Dimensionality | Concatenation with Body |
# |---------------|---------------------|----------------|------------------------|
# | **Title BoW** | Count vector using same `vocab.txt` | 5,634 sparse | Horizontal stack → 11,268 total |
# | **Title Unweighted FastText** | Average of token embeddings | 300 dense | Horizontal stack → 600 total |
# | **Title TF-IDF Weighted FastText** | TF-IDF weighted average of token embeddings | 300 dense | Horizontal stack → 600 total |
# 
# ### Design Choice — Title Preprocessing
# 
# The `clean_tokens()` function below applies the same core pipeline as Task 1 (regex tokenisation → lowercase → length filter → stopword removal → lemmatisation + stemming), but intentionally **omits** the corpus-level frequency filters (term-frequency and document-frequency filtering). This is a deliberate decision for two reasons:
# 
# 1. **For BoW (count vectors):** Title tokens are counted only if they exist in `vocab.txt` (built in Task 1 after all filtering). Tokens that would have been removed by frequency filtering are automatically excluded by the vocabulary restriction — the filter is implicitly applied without needing to recompute corpus statistics on titles alone.
# 
# 2. **For FastText embeddings:** We intentionally retain all cleaned title tokens (including rare ones) because FastText benefits from every available token — even a rare word contributes meaningful semantic information via its pretrained embedding. Discarding rare title tokens would reduce the already-short title representation (median 2 tokens) to near-empty vectors, degrading signal quality. The frequency filters in Task 1 were designed for the much longer review body (~7 tokens post-filtering); applying the same thresholds to 2-token titles would be overly aggressive.
# 
# **Alternatives considered:**
# - **Apply full Task 1 pipeline (including frequency filters) to titles:** Would remove too many tokens from already-short titles (average 2 tokens post-cleaning). Rejected.
# - **Use a title-specific vocabulary:** Would decouple title and body feature spaces, complicating concatenation for BoW. Rejected.
# - **Use raw (uncleaned) titles:** Would introduce inconsistency with the preprocessed review body. Rejected.
# - **Skip titles entirely:** Would miss the opportunity to test whether titles provide any lift. The assignment requires investigating this.
# 
# ### Design Choice — Concatenation Strategy (Not Averaging)
# 
# We **concatenate** title features alongside body features (horizontal stack) rather than averaging them together. This is important because:
# - Concatenation preserves the distinction between body and title signals — the model can learn different weights for body words vs title words.
# - Averaging would mix the two sources, losing the ability to weight them differently.
# - For BoW, the same word in the body and title occupies different column positions (columns 0–5633 for body, 5634–11267 for title), allowing the model to distinguish where the word appeared.

# %%
# 1a. Tokenise review titles using the same Task 1 cleaning (including stemming)
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

TOK_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
    STOPWORDS = set(line.strip().lower() for line in f if line.strip())

_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

def clean_tokens(text):
    """Same pipeline as Task 1: tokenise -> lowercase -> length filter -> stopwords -> lemmatise+stem."""
    if not isinstance(text, str):
        return []
    toks = [t.lower() for t in TOK_PATTERN.findall(text)]
    toks = [t for t in toks if len(t) >= 2 and t not in STOPWORDS]
    return [_stemmer.stem(_lemmatizer.lemmatize(t)) for t in toks]

df['title_tokens'] = df['review_title'].fillna('').apply(clean_tokens)
df['title_text']   = df['title_tokens'].apply(' '.join)
print(f"Median tokens per title: {df['title_tokens'].str.len().median():.0f}")
print(f"Empty-after-cleaning   : {(df['title_tokens'].str.len() == 0).sum()}")

# %%
# 1b. Title BoW — count vector aligned to vocab.txt
title_rows, title_cols, title_vals = [], [], []
for row_i, toks in enumerate(df['title_tokens']):
    cnt = Counter(t for t in toks if t in word_to_index)
    for w, c in cnt.items():
        title_rows.append(row_i); title_cols.append(word_to_index[w]); title_vals.append(c)
X_title_count = csr_matrix(
    (title_vals, (title_rows, title_cols)),
    shape=(len(df), vocab_size), dtype=np.float32,
)

# 1c. Title FastText embeddings (unweighted + TF-IDF weighted) — load FastText once
print("Loading FastText (cached locally) ...")
fasttext_model = gensim_api.load("fasttext-wiki-news-subwords-300")
print(f"FastText loaded. dim = {fasttext_model.vector_size}")

def avg_vector(tokens):
    vecs = [fasttext_model[t] for t in tokens if t in fasttext_model]
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else np.zeros(EMBED_DIM, dtype=np.float32)

X_title_unweighted = np.vstack([avg_vector(toks) for toks in df['title_tokens']])

# TF-IDF weights on the title corpus, then weighted-average FastText vectors
tfidf_title = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None)
tfidf_M     = tfidf_title.fit_transform(df['title_text'])
tfidf_vocab = {w: i for i, w in enumerate(tfidf_title.get_feature_names_out())}

X_title_weighted = np.zeros((len(df), EMBED_DIM), dtype=np.float32)
for row_i, toks in enumerate(df['title_tokens']):
    valid = [t for t in toks if t in fasttext_model]
    if not valid:
        continue
    vecs = np.array([fasttext_model[t] for t in valid], dtype=np.float32)
    weights = np.array(
        [tfidf_M[row_i, tfidf_vocab[t]] if t in tfidf_vocab else 0.0 for t in valid],
        dtype=np.float32,
    )
    s = weights.sum()
    X_title_weighted[row_i] = (vecs * weights[:, None]).sum(axis=0) / s if s > 0 else vecs.mean(axis=0)

print(f"X_title_count      : {X_title_count.shape}, nnz={X_title_count.nnz:,}")
print(f"X_title_unweighted : {X_title_unweighted.shape}")
print(f"X_title_weighted   : {X_title_weighted.shape}")

# %%
# 1d. Concatenate body + title for each representation -> Q2a inputs
X_q2a_count        = sp_hstack([X_count,       X_title_count]).tocsr()
X_q2a_unweighted   = np.hstack([X_unweighted,  X_title_unweighted])
X_q2a_weighted     = np.hstack([X_weighted,    X_title_weighted])

print(f"X_q2a_count       : {X_q2a_count.shape}")
print(f"X_q2a_unweighted  : {X_q2a_unweighted.shape}")
print(f"X_q2a_weighted    : {X_q2a_weighted.shape}")

# %% [markdown]
# ## Step 2 · Build the metadata features (Q2b)
# 
# ### Feature Engineering Decisions for Product Metadata
# 
# We add four metadata feature blocks to extend the text+title features. Each choice is justified below:
# 
# | Feature | Encoding | Dimensionality | Justification | Alternatives Considered |
# |---------|----------|----------------|---------------|------------------------|
# | **Brand name** | One-hot encoding | 11 columns (one per brand) | Only 11 distinct brands in the dataset — one-hot is feasible and avoids imposing arbitrary ordinal relationships between brands. The EDA (Task 1) showed brand-specific buyer rates vary from ~65% to ~90%, confirming brand carries predictive signal. | Label encoding — imposes false ordinal relationship (e.g., "L'Oreal" < "Nykaa" has no meaning); Target encoding — risks data leakage without careful CV-aware implementation; Dropped entirely — wastes known signal. |
# | **Product title** | Bag-of-Words (count vectors aligned to `vocab.txt`) | 5,634 columns (same vocabulary as review text) | Product titles contain descriptive terms (e.g., "serum", "sunscreen", "lipstick") that contextualise the review. Using the same vocabulary ensures consistency and avoids introducing new dimensions. Products with certain keywords may attract more/fewer verified buyers. | TF-IDF on product titles — possible but adds complexity; separate vocabulary — would increase dimensionality without clear benefit; Dropped — loses useful product-type signal. |
# | **Average product rating** | Z-score normalisation (StandardScaler) | 1 column | The EDA showed a weak but non-zero correlation (r = 0.044) with `is_a_buyer`. Z-scoring centres the feature (mean=0, std=1) so its scale does not dominate other features when concatenated with sparse count vectors. | Raw value — different scale from other features could bias distance-based splits; Min-max scaling — sensitive to outliers; Binning — loses granularity; Dropped — wastes available signal. |
# | **log(price + 1)** | Log-transform then Z-score | 1 column | `price` had the strongest correlation with `is_a_buyer` (r = −0.206) in the EDA. The log transform is applied because price distributions in e-commerce are typically right-skewed (few expensive products, many cheap ones) — log compression reduces skewness and makes the feature more normally distributed, which benefits linear models. The `+1` prevents `log(0)` for any zero-price items. Z-scoring after log ensures scale compatibility. | Raw price — highly skewed, dominated by outliers; Sqrt transform — less effective at reducing skewness than log for multiplicative data; Binning into price ranges — loses continuous signal; Dropped — wastes the strongest structured predictor. |
# 
# ### Why these 4 features and not others?
# 
# **Included:**
# - The EDA (Task 1) identified `price` and `product_rating_count` as the two strongest structured predictors. We use `price` (log-transformed) and `avg_product_rating` as they provide complementary information about product quality/affordability.
# - `brand_name` showed variable buyer rates across brands (Task 1 EDA bar chart).
# - `product_title` provides categorical product-type information in text form.
# 
# **Excluded:**
# - **`product_rating_count`:** While it showed moderate correlation (r = 0.174), it is highly collinear with `avg_product_rating` and `product_encoded` (from EDA). Including it would add redundancy and potential multicollinearity.
# - **`review_rating`:** Negligible correlation (r = 0.029) with `is_a_buyer` — confirmed in EDA as non-discriminative.
# - **`review_date`/`year`:** Temporal analysis in Task 1 showed stable buyer proportions across years — no temporal signal to exploit.
# - **`product_id`/`review_id`:** Identifiers with no semantic meaning.
# - **`author`:** Too many unique values (~50K), leading to extreme sparsity if one-hot encoded; and reviewer identity should not generalise to unseen reviewers.
# - **`product_tags`:** 78% missing values — imputation would introduce excessive noise.
# - **`product_url`:** Contains no information beyond what `product_title` and `brand_name` already capture.
# 
# ### Concatenation Strategy
# 
# All metadata features are concatenated into a single sparse block `X_meta`, then horizontally stacked with the text+title features. This "wide" concatenation preserves the original text features while simply appending the new dimensions — the model can then learn which features (text vs metadata) carry signal.

# %%
# 2a. Brand — one-hot (only 11 distinct brands so we keep them all)
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
X_brand = ohe.fit_transform(df[['brand_name']])

# 2b. Product title — clean and count-vectorise against vocab.txt
df['product_title_clean'] = df['product_title'].fillna('').apply(lambda s: ' '.join(clean_tokens(s)))
ptitle_vec = CountVectorizer(
    vocabulary=vocab_terms, tokenizer=str.split, lowercase=False, token_pattern=None,
)
X_ptitle = ptitle_vec.fit_transform(df['product_title_clean']).astype(np.float32)

# 2c. Numeric — avg_product_rating + log(price+1), z-scored
numeric = pd.DataFrame({
    'avg_product_rating': df['avg_product_rating'].fillna(df['avg_product_rating'].median()),
    'log_price'         : np.log1p(df['price'].fillna(df['price'].median())),
}).to_numpy(dtype=np.float32)
X_numeric = StandardScaler().fit_transform(numeric).astype(np.float32)

# 2d. Combined metadata block
X_meta = sp_hstack([X_brand, X_ptitle, csr_matrix(X_numeric)]).tocsr()
print(f"X_meta : {X_meta.shape}, nnz={X_meta.nnz:,}")

# Q2b inputs — body + title + metadata
X_q2b_count       = sp_hstack([X_q2a_count,                  X_meta]).tocsr()
X_q2b_unweighted  = sp_hstack([csr_matrix(X_q2a_unweighted), X_meta]).tocsr()
X_q2b_weighted    = sp_hstack([csr_matrix(X_q2a_weighted),   X_meta]).tocsr()
print(f"X_q2b_count       : {X_q2b_count.shape}")
print(f"X_q2b_unweighted  : {X_q2b_unweighted.shape}")
print(f"X_q2b_weighted    : {X_q2b_weighted.shape}")

# %% [markdown]
# ## Step 3 · Run Q2 — Progressive feature addition (body → +title → +metadata)
# 
# ### Experimental Design for Q2
# 
# **Research question:** Does adding structured information (review title, product metadata) to text features improve classification performance?
# 
# **Methodology — Progressive comparison:**
# 
# We evaluate **three scopes** of increasing information richness, keeping the same 3 representations × 3 models throughout:
# 
# | Scope | Features Included | Dimensionality (BoW) | Rationale |
# |-------|-------------------|---------------------|-----------|
# | **Body only** (Q1 baseline) | Review text features | 5,634 | Baseline — text signal alone |
# | **Body + Title** (Q2a) | Review text + review title | 11,268 (5,634×2) | Tests whether the short review title provides complementary signal beyond the review body |
# | **Body + Title + Metadata** (Q2b) | Review text + title + brand + product title + rating + price | ~16,915 | Tests whether structured product/price information provides orthogonal signal |
# 
# **Why progressive addition (not ablation)?**
# - Progressive addition mirrors a realistic deployment scenario: you start with the minimum viable feature set (text) and incrementally invest effort in feature engineering. Each step answers "is the engineering effort worth the accuracy gain?"
# - An ablation study (start with everything, remove one at a time) would answer a different question ("which feature is most important?") — less actionable for this milestone.
# 
# **Why 27 total evaluations (9 per scope)?**
# - Running all 3 representations × 3 models at each scope ensures we can detect **interactions** between feature type and model type. For example, metadata might help Logistic Regression more than Random Forest (because metadata provides easy linear shortcuts), or BoW might benefit more from title addition than embeddings do.
# 
# **Control:** Same CV splits, same random seed, same evaluation function, same hyperparameters. The ONLY variable changing between Q1, Q2a, and Q2b is the feature matrix input.
# 
# For comparability, we re-tag the Q1 (body-only) results rather than re-running them, ensuring perfectly consistent numbers.

# %%
# 3a. Bundle every (scope, representation) pair into one dict and loop
ALL_INPUTS = {
    ('Body only', 'Count (BoW)')              : X_count,
    ('Body only', 'Unweighted FastText')      : X_unweighted,
    ('Body only', 'TF-IDF Weighted FastText') : X_weighted,
    ('Body + Title', 'Count (BoW)')              : X_q2a_count,
    ('Body + Title', 'Unweighted FastText')      : X_q2a_unweighted,
    ('Body + Title', 'TF-IDF Weighted FastText') : X_q2a_weighted,
    ('Body + Title + Metadata', 'Count (BoW)')              : X_q2b_count,
    ('Body + Title + Metadata', 'Unweighted FastText')      : X_q2b_unweighted,
    ('Body + Title + Metadata', 'TF-IDF Weighted FastText') : X_q2b_weighted,
}

# 3b. We already evaluated 'Body only' in Q1 — reuse those rows. Run only Q2a and Q2b here.
results_q2 = list(q1_df.assign(Scope='Body only').to_dict('records'))

for (scope, rep_name), X in ALL_INPUTS.items():
    if scope == 'Body only':
        continue   # already done in Part 1
    for model_name, make_model in MODELS.items():
        print(f"Evaluating  {scope:<25s} | {rep_name:<28s} | {model_name}")
        metrics = evaluate(X, y, make_model())
        results_q2.append({
            'Scope'          : scope,
            'Representation' : rep_name,
            'Model'          : model_name,
            **metrics,
        })

all_df = pd.DataFrame(results_q2)
print(f"\nTotal evaluations: {len(all_df)}")

# %% [markdown]
# ## Step 4 · Q2 Analysis — Measure the lift from additional features
# 
# ### Comparison Methodology
# 
# We present the Q2 results using four complementary views:
# 1. **Full results table** — all 27 configurations sorted by Macro-F1, enabling direct comparison across scopes.
# 2. **Pivot table** — Macro-F1 organised by Scope × (Representation, Model), showing the progressive improvement.
# 3. **Lift line chart** — one panel per representation, showing how Macro-F1 changes as features are added. Lines connect the same model across scopes, making it easy to see which models benefit most from additional features.
# 4. **Best overall configuration** — identifying the single best-performing combination for potential deployment.
# 
# **Key question to answer:** Is the lift from adding title/metadata consistent across all representations and models, or does it interact with the learning algorithm?

# %%
# 4a. Combined table — all 27 configurations sorted by Macro-F1
print("=== ALL RESULTS (sorted by Macro-F1) ===")
cols = ['Scope','Representation','Model','Accuracy','Macro-Precision','Macro-Recall','Macro-F1','ROC-AUC']
print(all_df[cols].sort_values('Macro-F1', ascending=False).round(4).to_string(index=False))

# %%
# 4b. Pivot — Macro-F1 by Scope x (Representation, Model)
pivot_f1 = all_df.pivot_table(
    index='Scope',
    columns=['Representation', 'Model'],
    values='Macro-F1',
).reindex(['Body only', 'Body + Title', 'Body + Title + Metadata'])

print("Macro-F1 across all scopes / representations / models:")
print(pivot_f1.round(4).to_string())

# %%
# 4c. Lift chart — Macro-F1 across scopes, one line per (representation, model)
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

scope_order = ['Body only', 'Body + Title', 'Body + Title + Metadata']
reps        = ['Count (BoW)', 'Unweighted FastText', 'TF-IDF Weighted FastText']
model_color = {'Logistic Regression': '#3b7dd8',
               'Decision Tree'      : '#e08e3a',
               'Random Forest'      : '#5fb56b'}

for ax, rep in zip(axes, reps):
    for model_name, color in model_color.items():
        ys = [
            all_df.query(
                "Scope==@s and Representation==@rep and Model==@model_name"
            )['Macro-F1'].iloc[0]
            for s in scope_order
        ]
        ax.plot(scope_order, ys, marker='o', linewidth=2, label=model_name, color=color)
        for x_, y_ in zip(scope_order, ys):
            ax.text(x_, y_+0.008, f"{y_:.3f}", ha='center', fontsize=8, color=color)
    ax.set_title(rep)
    ax.set_ylabel('Macro-F1')
    ax.set_ylim(0.40, 0.78)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

plt.suptitle('Q2 lift: Macro-F1 as we add information (body -> +title -> +metadata)', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %%
# 4d. Best overall configuration across the 27 cells
best = all_df.loc[all_df['Macro-F1'].idxmax()]
print("Best overall configuration (by Macro-F1):")
print(f"  Scope          : {best['Scope']}")
print(f"  Representation : {best['Representation']}")
print(f"  Model          : {best['Model']}")
print(f"  Accuracy        = {best['Accuracy']:.4f}")
print(f"  Macro-Precision = {best['Macro-Precision']:.4f}")
print(f"  Macro-Recall    = {best['Macro-Recall']:.4f}")
print(f"  Macro-F1        = {best['Macro-F1']:.4f}")
print(f"  ROC-AUC         = {best['ROC-AUC']:.4f}")

# %% [markdown]
# ---
# ## Findings and Discussion
# 
# ### Q1 — Which feature representation performs best (review text only)?
# 
# #### Key Results
# 
# | Rank | Representation | Best Model | Macro-F1 | ROC-AUC |
# |------|---------------|------------|----------|---------|
# | 1 | Count (BoW) | Random Forest | 0.5801 | 0.6742 |
# | 2 | Count (BoW) | Decision Tree | 0.5781 | 0.5850 |
# | 3 | Count (BoW) | Logistic Regression | 0.5564 | 0.6908 |
# | 4 | Unweighted FastText | Decision Tree | 0.5493 | 0.5481 |
# | 5 | TF-IDF Weighted FastText | Decision Tree | 0.5444 | 0.5439 |
# 
# #### Analysis
# 
# 1. **Count (BoW) dominates all embedding representations on Macro-F1.** This confirms our hypothesis: for predicting `is_a_buyer`, the *identity* of specific words (e.g., "bought", "deliver", "return", "receiv") matters more than general semantic similarity. BoW preserves exactly this information — a sparse vector where each dimension corresponds to a specific vocabulary word and its count.
# 
# 2. **Why embeddings underperform BoW here:**
#    - **Information loss through averaging:** When we average 300-d vectors across all tokens in a review, individual word identities are "washed out." A review containing "bought" and a review containing "considering" might produce similar average vectors if their other words are semantically close — but "bought" is a strong buyer signal that BoW preserves as a distinct dimension.
#    - **Dimensionality mismatch:** BoW has 5,634 dimensions (one per word) — each is a clean binary/count signal. Embeddings compress this into 300 continuous dimensions, losing the discrete word-presence information that drives this classification task.
#    - **TF-IDF weighting provides minimal benefit over unweighted:** Both embedding variants perform nearly identically (F1 difference < 0.01). This suggests that the TF-IDF scores do not meaningfully distinguish discriminative tokens for this particular classification task — the averaging process still dominates.
# 
# 3. **Model comparison (within BoW):**
#    - **Random Forest** achieves the best Macro-F1 (0.5801) because it can capture feature interactions (e.g., "bought" AND "delivered" together are more predictive than either alone) through tree splits.
#    - **Logistic Regression** has the best ROC-AUC (0.6908) despite lower Macro-F1 — its probability calibration is better than the forest's, meaning its continuous predictions rank samples more accurately even if the default 0.5 threshold is suboptimal for the imbalanced classes.
#    - **Decision Tree** is surprisingly competitive on F1 (0.5781) — the fully-grown tree memorises training patterns effectively, and the 5-fold CV still generalises reasonably because BoW features are individually clean signals.
# 
# 4. **All models struggle with the minority class (non-buyers, 20.3%):** Even the best Macro-F1 is only 0.58, indicating that text features alone provide modest — not strong — discriminative signal for buyer status. This motivates Q2's addition of structured features.
# 
# ---
# 
# ### Q2 — Does adding more information improve accuracy?
# 
# #### Progressive Lift Summary (Macro-F1, best model per scope)
# 
# | Scope | Best Config | Macro-F1 | Δ vs Body Only |
# |-------|-------------|----------|----------------|
# | Body only | BoW + Random Forest | 0.5801 | — |
# | Body + Title | BoW + Logistic Regression | 0.5766 | −0.0035 |
# | Body + Title + Metadata | BoW + Random Forest | **0.7117** | **+0.1316** |
# 
# #### Detailed Analysis
# 
# **1. Adding review title (Q2a) provides negligible or mixed lift.**
# 
# - For BoW, adding the title does not improve and in some cases slightly decreases performance (−0.004 for Random Forest). This makes sense: review titles are very short (median 2 tokens) and often repeat words already present in the review body. The additional 5,634 sparse dimensions (title BoW) mostly contain zeros, adding noise without new signal.
# - For FastText embeddings, the title similarly provides minimal lift. The 300 additional embedding dimensions from a 2-token title are dominated by noise — with so few tokens, the averaged vector is unstable and unreliable.
# - **Conclusion:** Review titles are too short and too redundant with the review body to provide meaningful additional classification signal for `is_a_buyer`.
# 
# **2. Adding metadata (Q2b) provides a dramatic lift — the largest single improvement in the pipeline.**
# 
# - Macro-F1 jumps from ~0.58 to **0.71** (+0.13 absolute, +23% relative) — the single largest improvement across all experiments.
# - ROC-AUC jumps from ~0.67 to **0.88** — an enormous improvement in ranking quality.
# - This confirms the EDA finding: while individual structured features have weak linear correlations with `is_a_buyer` (max |r| = 0.206), they provide **strong non-linear signal** when combined with text features. The forest/logistic models can exploit interactions between price, brand, and text that individual correlation analysis cannot detect.
# 
# **3. Model convergence with metadata:**
# - With metadata added, Logistic Regression (0.7069) nearly matches Random Forest (0.7117) — a gap of only 0.005. This suggests the metadata provides near-linear separability that even a simple linear model can exploit.
# - Decision Tree also dramatically improves (from 0.578 to 0.691), confirming that the metadata features provide easy, clean splits.
# 
# **4. Representation differences shrink with metadata:**
# - Without metadata: BoW leads embeddings by ~0.03–0.06 F1.
# - With metadata: the gap narrows to ~0.02. When strong structured features dominate the signal, the text representation matters less — metadata carries the heavy lifting.
# 
# ---
# 
# ### Practical Recommendations
# 
# 1. **Deployment recommendation:** `Body + Title + Metadata` with `Count (BoW)` + `Random Forest` (Macro-F1 = 0.7117, ROC-AUC = 0.8762). This is the best overall configuration.
# 
# 2. **If interpretability is required:** Use `Logistic Regression` at the same scope (Macro-F1 = 0.7069, ROC-AUC = 0.8767). The per-feature coefficients are directly inspectable, and the ROC-AUC is actually marginally higher.
# 
# 3. **If only text is available** (no metadata): Use `Count (BoW)` + `Random Forest` (Macro-F1 = 0.5801). Embeddings do not outperform simple word counts for this task.
# 
# 4. **Key insight for Milestone 2:** The single most impactful feature engineering step is adding product metadata (especially price and brand). Future work should focus on enriching metadata features (e.g., review length, time-since-product-launch) rather than improving text representations.
# 
# ---
# 
# ### Limitations
# 
# 1. **No hyperparameter tuning:** All models use default parameters. Tuned models (especially gradient-boosted trees) could likely improve absolute performance, but the relative ranking of representations would likely be preserved.
# 2. **No feature selection:** We concatenate all features without pruning redundant dimensions. L1-regularised models or feature importance thresholding could reduce dimensionality.
# 3. **Class imbalance not addressed directly:** We did not apply SMOTE, class weighting, or threshold tuning. These techniques could improve minority-class recall at the cost of majority-class precision.
# 4. **Static embeddings only:** Contextual embeddings (BERT, RoBERTa) could capture nuances that static FastText vectors miss, but are out of scope for this milestone.

# %% [markdown]
# ---
# ## References
# 
# 1. **Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T.** (2017). Enriching Word Vectors with Subword Information. *Transactions of the Association for Computational Linguistics*, 5, 135–146. https://doi.org/10.1162/tacl_a_00051
#    - The FastText embedding model (`fasttext-wiki-news-subwords-300`) used for unweighted and TF-IDF weighted representations.
# 
# 2. **Mikolov, T., Chen, K., Corrado, G., & Dean, J.** (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv preprint arXiv:1301.3781*.
#    - Word2Vec — the alternative embedding model considered but rejected due to lack of subword/OOV handling.
# 
# 3. **Pennington, J., Socher, R., & Manning, C. D.** (2014). GloVe: Global Vectors for Word Representation. *Proceedings of EMNLP*, 1532–1543.
#    - GloVe — another alternative embedding model considered but rejected for the same reason.
# 
# 4. **Salton, G. & Buckley, C.** (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
#    - Foundation for TF-IDF weighting used in the weighted FastText representation.
# 
# 5. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
#    - Reference for cross-validation methodology (K-fold, stratified), bias-variance tradeoff, and evaluation protocols.
# 
# 6. **Fawcett, T.** (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
#    - Foundation for the ROC-AUC evaluation metric.
# 
# 7. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press.
#    - Reference for bag-of-words representation, TF-IDF, and text classification fundamentals.

# %% [markdown]
# ---
# ## Individual Contributions
# 
# | Member | Student ID | Contribution |
# |--------|-----------|--------------|
# | **Vo Ngoc Dung** | S4124370 | Led **Task 1 (Preprocessing & EDA)**. Designed and implemented the full text preprocessing pipeline: regex tokenisation, lowercase normalisation, stopword removal (custom list), lemmatisation + stemming, and corpus-level frequency filtering (TF/DF thresholds). Produced the cleaned `processed.csv`, `vocab.txt`, and `stopwords_en.txt` used by all downstream tasks. Conducted the exploratory data analysis including class distribution, correlation analysis, and brand-level buyer rate comparisons. |
# | **Tang Hoang Ha** | S4147768 | Led **Task 3 (Classification)**. Designed the full experimental framework: classifier selection rationale (Logistic Regression, Decision Tree, Random Forest), evaluation protocol (5-fold stratified CV with 5 metrics), and Macro-F1 as primary metric justification. Implemented the Q1 pipeline (9 representation × model evaluations), Q2a (title feature engineering with BoW and FastText concatenation), and Q2b (metadata feature engineering — brand one-hot, log-price, product title BoW, scaled rating). Wrote all analysis, discussion, pivot tables, visualisations, and the findings section. |
# | **Nguyen Quoc Trong Nghia** | S3343711 | Contributed to **Task 2 (Feature Representation)** and **Task 3 (Classification)**. Helped design the three feature representations (Count BoW, Unweighted FastText, TF-IDF Weighted FastText). Assisted with the Q2 experimental design — particularly the progressive feature addition strategy and metadata feature selection decisions. Participated in results interpretation and the discussion of why metadata provides the largest performance lift (+0.13 Macro-F1). |
# | **Nguyen Anh Duc** | S4136756 | Contributed to **Task 2 (Feature Representation)** and **Task 3 (Classification)**. Implemented the FastText embedding pipeline (unweighted averaging and TF-IDF weighted averaging using `fasttext-wiki-news-subwords-300`). Assisted with the title feature engineering in Q2a (tokenisation, vocabulary alignment, embedding computation). Participated in the evaluation protocol design and comparative analysis across all 27 configurations. |
# 
# ### Collaboration Summary
# 
# - **Task 1 (Preprocessing & EDA):** Dung — primary author; outputs consumed by all subsequent tasks.
# - **Task 2 (Feature Representation):** Nghia and Duc — built the three vector representations (BoW, unweighted FastText, TF-IDF weighted FastText).
# - **Task 3 (Classification):** Ha — led pipeline design, implementation, and write-up; Nghia and Duc contributed to experimental design, feature engineering, and analysis.
# - **Coordination:** The team used shared files (`processed.csv`, `vocab.txt`, `count_vectors.txt`, embedding files) as the interface between tasks, ensuring reproducibility and consistency across all three milestones.


