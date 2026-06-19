# MOPSO-Based CNN for Keyword Selection on Google Ads

> B.E. Final Year Project — Department of Information Science & Engineering  
> Adichunchanagiri Institute of Technology, VTU (2020–21)  
> **IISc Bangalore "Best Project of the Year" — 44th Student Project Programme**

---

## Overview

Google Ads presents advertisers with thousands of potential keywords. Manually selecting the right ones is slow and inaccurate. This project builds an intelligent keyword selection system that automatically identifies the most effective keywords for a given ad campaign — achieving **93.85% prediction accuracy**.

The core innovation is combining a **Convolutional Neural Network (CNN)** for text classification with **Multi-Objective Particle Swarm Optimisation (MOPSO)** to automatically find the optimal CNN architecture — improving both accuracy and training speed simultaneously.

---

## The Problem

Three key challenges in keyword selection:

| Challenge | Our Solution |
|---|---|
| Keywords in multiple languages (Chinese/English) | Multilingual corpus selection + GloVe word embeddings |
| Heavily imbalanced dataset (few good keywords, many bad ones) | SMOTE / Borderline-SMOTE resampling |
| CNN architecture search is manual and time-consuming | MOPSO automates neural architecture search |

---

## How It Works

### Pipeline

```
Google Ads Keywords
        ↓
[1] Corpus Selection        → Select multilingual corpus (Wikipedia, Weibo, Baidu)
        ↓
[2] Word Embedding          → Convert keywords to vectors (Word2Vec / GloVe / FastText)
        ↓
[3] Imbalanced Data Handling → SMOTE resampling to balance classes
        ↓
[4] Improved CNN            → Classify keywords (positive / negative match)
        ↓
[5] MOPSO Optimisation      → Find best CNN parameters automatically
        ↓
Output: Ranked positive & negative keywords for the advertiser
```

### Key Components

**Word Embedding**
Keywords are converted to dense vector representations using GloVe embeddings, capturing semantic meaning and relationships between words across languages.

**Class Imbalance Handling**
The dataset is heavily skewed — very few keywords are truly relevant. We applied SMOTE (Synthetic Minority Oversampling Technique) and Borderline-SMOTE to balance training data.

**Improved CNN**
A modified CNN that extracts both linguistic features and contextual keyword features — going beyond standard text CNNs which only capture language characteristics.

**MOPSO (Multi-Objective Particle Swarm Optimisation)**
MOPSO treats CNN hyperparameter selection as a multi-objective optimisation problem — simultaneously minimising training time and maximising classification accuracy. Each "particle" represents a CNN configuration; the swarm converges on the optimal architecture.

---

## Results

| Metric | Value |
|---|---|
| Keyword prediction accuracy | **93.85%** |
| CNN accuracy (before MOPSO) | ~21.9% |
| CNN accuracy (after MOPSO optimisation) | ~29.4% |
| Best model saved | `weights.002-0.6587.hdf5` |
| Categories classified | Cosmetics, Electronics, Furniture, Clothing |

MOPSO reduced training time while improving validation accuracy — the core multi-objective result.

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| ML/DL | TensorFlow, Keras, scikit-learn |
| NLP | GloVe, Word2Vec, FastText, NLTK |
| Optimisation | Custom MOPSO implementation |
| Data handling | SMOTE (imbalanced-learn), pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Web interface | Flask |
| IDE | Jupyter Notebook, Anaconda |

---

## Project Structure

```
mopso-cnn-keyword-selection/
│
├── keyword_model.py          # Core CNN model definition
├── mopso_optimizer.py        # MOPSO implementation for CNN architecture search
├── data_preprocessing.py     # Word embedding, corpus selection, SMOTE
├── train.py                  # Training pipeline
├── predict.py                # Keyword prediction (positive/negative classification)
├── app.py                    # Flask web application
├── requirements.txt          # Dependencies
└── README.md
```

---

## Setup & Usage

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Train the model**
```bash
python train.py
```

**Run MOPSO optimisation**
```bash
python mopso_optimizer.py
```

**Start the web app**
```bash
python app.py
```
Then visit `http://localhost:5000` — enter a keyword to get positive/negative classification with confidence score.

**Run prediction directly**
```bash
python predict.py --keyword "buy electronics online"
```

---

## Example Output

```
Input keyword: "digital electronics seller"

Suggested domain: Electronics
Accuracy: 93.856858%

Positive Keywords: ['best buy electronics', 'technology store', 'cheap electronics', 
                    'electronics store near me']
Negative Keywords: ['free electronics repair', 'electronics recycling']
```

---

## Academic Context

- **Institution:** Adichunchanagiri Institute of Technology, Chikkamagaluru (VTU affiliated)
- **Award:** Best Project of the Year — 44th Student Project Programme, IISc Bangalore (2020–21)
- **Guide:** Mrs. Anjali, M.Tech, Assistant Professor, Dept. of IS&E
- **Team:** Nisha K Gowda, Nithyashree B.L, Noorul Huda, Pranamya Kashyap M.P

---

## References

Key papers this work builds on:
- Szymanski & Lipinski (2018) — Model of effectiveness of Google AdWords
- Sun, Xue & Zhang (2018) — PSO-based flexible CNN for image classification
- Fielding & Zhang (2018) — Evolving image classification architectures with enhanced PSO
- Barua et al. (2014) — MWMOTE for imbalanced dataset learning
