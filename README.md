# Yelp Restaurant & Hotel Sentiment Analysis with DistilBERT

A comprehensive sentiment analysis project that classifies Yelp restaurant and hotel reviews into positive, negative, and neutral sentiments using DistilBERT transformer model.

## 🎯 Project Overview

This project implements a deep learning approach to sentiment analysis using:
- **Dataset**: Yelp restaurant and hotel reviews
- **Model**: DistilBERT (Distilled Bidirectional Encoder Representations from Transformers)
- **Task**: Multi-class text classification (positive, negative, neutral)
- **Framework**: PyTorch with Hugging Face Transformers

## 📁 Project Structure

```
yelp-distilbert-sentiment-analysis/
├── data/
│   └── yelp_restaurants_hotels_ver2.csv     # Raw dataset
├── notebooks/
│   ├── 01_data_processing_and_training.ipynb # Main training notebook
│   └── 02_model_inference.ipynb             # Inference and evaluation
├── utils/
│   └── data_preprocessing_utils.ipynb       # Utility functions
├── models/
│   └── best_distilbert_model.pt            # Trained model weights
├── outputs/
│   └── (generated during training)          # Logs, plots, processed data
├── requirements.txt                         # Python dependencies
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yelp-distilbert-sentiment-analysis.git
   cd yelp-distilbert-sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Place `yelp_restaurants_hotels_ver2.csv` in the `data/` directory

### Usage

#### 1. Data Preprocessing
Use the utility notebook to prepare your data:
```python
# In utils/data_preprocessing_utils.ipynb
from data_preprocessing_utils import process_yelp_data

# Process the complete pipeline
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, processor = process_yelp_data("../data/yelp_restaurants_hotels_ver2.csv")
```

#### 2. Model Training
Run the training notebook:
- Open `notebooks/01_data_processing_and_training.ipynb`
- Execute all cells to train the DistilBERT model
- Model will be saved to `models/best_distilbert_model.pt`

#### 3. Model Inference
Use the inference notebook:
- Open `notebooks/02_model_inference.ipynb`
- Load the trained model and make predictions
- Evaluate model performance on test data

## 📊 Dataset Information

- **Source**: Yelp Open Dataset
- **Content**: Restaurant and hotel reviews
- **Labels**: 3-class sentiment (positive, negative, neutral)
- **Features**: 
  - `cleaned_text`: Preprocessed review text
  - `star_sentiment`: Sentiment labels based on star ratings
  - Additional metadata (business info, user info, etc.)

### Data Distribution
- **Positive**: Reviews with 4-5 stars
- **Negative**: Reviews with 1-2 stars  
- **Neutral**: Reviews with 3 stars

## 🤖 Model Architecture

**DistilBERT** (Distilled BERT):
- 6 transformer layers (vs 12 in BERT-base)
- 66M parameters (vs 110M in BERT-base)
- ~60% smaller and faster than BERT
- Retains 97% of BERT's performance

**Classification Head**:
- DistilBERT backbone (pre-trained)
- Dropout layer (0.3)
- Linear layer (768 → 3 classes)

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | ~85-90% |
| F1-Score (Macro) | ~83-88% |
| Training Time | ~2-3 hours (GPU) |

*Note: Exact scores depend on dataset size and training parameters*

## 🛠 Technical Details

### Key Components

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Label encoding (text → numerical)
   - Train/validation/test splitting (70/15/15)

2. **Model Training**
   - Tokenization with DistilBERT tokenizer
   - Gradient accumulation for batch processing
   - Learning rate scheduling
   - Early stopping based on validation loss

3. **Evaluation**
   - Confusion matrix analysis
   - Per-class precision, recall, F1-score
   - ROC curves and AUC scores

### Hyperparameters
- **Learning Rate**: 2e-5
- **Batch Size**: 16 (with gradient accumulation)
- **Epochs**: 3-5
- **Max Sequence Length**: 512 tokens
- **Optimizer**: AdamW
- **Scheduler**: Linear with warmup

## 📋 Requirements

```
torch>=1.9.0
transformers>=4.20.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
jupyter>=1.0.0
```

## 🔄 Workflow

1. **Data Loading** → Load Yelp reviews from CSV
2. **Preprocessing** → Clean text, encode labels, split data
3. **Tokenization** → Convert text to DistilBERT tokens
4. **Training** → Fine-tune DistilBERT on sentiment task
5. **Evaluation** → Test on held-out data
6. **Inference** → Predict sentiment for new reviews

## 📸 Results & Visualizations

The notebooks generate various visualizations:
- Training/validation loss curves
- Confusion matrices
- Label distribution plots
- Sample predictions with confidence scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **Yelp** for providing the open dataset
- **Google Research** for the original BERT/DistilBERT models

## 📞 Contact

For questions or suggestions:
- Create an issue in this repository
- Email: [your-email@domain.com]

## 🔗 References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Yelp Open Dataset](https://www.yelp.com/dataset)

---

**⭐ If you find this project helpful, please give it a star!**
