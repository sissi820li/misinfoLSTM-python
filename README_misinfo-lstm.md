# MisinfoLSTM: Medical Misinformation Detection with a Bidirectional LSTM

A PyTorch text classifier that labels short health claims and tweets as **true** or **fake**. It was trained on about 1,600 labelled medical and COVID-19 claims, then checked on a separate set of claims it never saw during training.

## Results

| Model | Evaluation set | Accuracy |
|---|---|---|
| MisinfoLSTM (bidirectional LSTM) | Test split, 244 claims | **97.95%** (precision and recall 0.97–0.99 on both classes) |
| MisinfoLSTM (bidirectional LSTM) | Unseen CoAID claims, 27 claims from a different source | **96.30%** (26 of 27) |
| Baseline: TF-IDF + logistic regression | Test split of the training data, 171 claims | 97.66% |

Example prediction:

```
Claim:      "Drinking water cures COVID"
Prediction: Fake  (confidence 0.9998)
```

## How it works

**Data**
- Two labelled sources were merged for training, validation and testing: a set of realistic medical misinformation claims, and the ACOVMD set of COVID-19 tweets. That's 1,623 English claims after filtering.
- Claims from the CoAID dataset were kept completely separate as a final check on new data.
- The split was 70% training, 15% validation and 15% test.

**Preprocessing**
- Lowercase the text and remove URLs, hashtags and non-letter characters.
- Remove English stop words (NLTK).
- Build a vocabulary from the training set only, with a padding token (0) and an unknown-word token (1).
- Pad or cut each claim to 50 tokens.

**Model**
- Embedding layer with 100 dimensions.
- 2-layer bidirectional LSTM with 64 hidden units per direction.
- The final forward and backward hidden states are joined, passed through a 20% dropout layer, and fed into a single linear output.
- Trained with `BCEWithLogitsLoss`, using Adam (learning rate 0.001, weight decay 1e-4), batch size 32, for 30 epochs.

**Changes from the first prototype**
1. Lowered the learning rate from 0.01 to 0.001. The prototype was overshooting.
2. Added 20% dropout, so the model can't depend on a few "trigger" words.
3. Added weight decay (L2 regularization), to keep weights small so the model generalizes to new claims.

## Repository layout

```
clean_dataset.ipynb     # merge sources, filter non-English text, clean, tokenize, build vocab, split, save tensors
baseline_model.ipynb    # TF-IDF + logistic regression baseline
Model_prototype.ipynb   # first LSTM version
Model_final.ipynb       # final MisinfoLSTM: training, evaluation, metrics, sample predictions
dataset/                # cleaned CSV splits, tensor splits, vocab_to_int.json
```

## Running it

```bash
pip install torch numpy pandas scikit-learn nltk matplotlib openpyxl
python -c "import nltk; nltk.download('stopwords')"
```

Run `clean_dataset.ipynb` first to build the tensors in `dataset/`. Then run `baseline_model.ipynb` and `Model_final.ipynb`.

## Limitations

- The unseen CoAID set is small (27 claims). Read the 96.3% as a sanity check, not a strong benchmark.
- On the main test split, a simple TF-IDF baseline also scores very high. These datasets are fairly easy to separate by word choice. The harder question is whether the model holds up on carefully worded misinformation.
- English only. Claims are capped at 50 tokens.

## Tools

Python, PyTorch, NumPy, Pandas, scikit-learn, NLTK, Matplotlib, Jupyter
