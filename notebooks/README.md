# Notebooks Overview

This directory contains the main Jupyter notebooks for the Yelp sentiment analysis project using DistilBERT.

## � Notebook Structure

### 1. `01_data_processing_and_training.ipynb`
**Main training notebook** - Complete end-to-end pipeline for training the DistilBERT model.

**Contents:**
- Data loading and exploration
- Text preprocessing and cleaning
- Label encoding and data splitting
- DistilBERT model setup and configuration
- Training loop with validation
- Model evaluation and metrics
- Saving the trained model

**Expected Runtime:** 2-3 hours with GPU, 6-8 hours with CPU

### 2. `02_model_inference.ipynb`
**Inference and evaluation notebook** - Load trained model and perform predictions.

**Contents:**
- Loading the pre-trained DistilBERT model
- Processing new text inputs
- Single and batch predictions
- Model performance analysis
- Confusion matrix and classification reports
- Example predictions with confidence scores

**Expected Runtime:** 10-20 minutes

## 🚀 Getting Started

### Prerequisites
Before running the notebooks, ensure you have:
1. Installed all requirements: `pip install -r ../requirements.txt`
2. Downloaded the dataset: `../data/yelp_restaurants_hotels_ver2.csv`
3. CUDA-compatible GPU (recommended) or sufficient RAM for CPU training

### Running Order
1. **First Time Setup:**
   - Run `../utils/data_preprocessing_utils.ipynb` to understand the data processing functions
   - Then run `01_data_processing_and_training.ipynb` for complete training

2. **Inference Only:**
   - If you have a pre-trained model, directly run `02_model_inference.ipynb`

### Expected Outputs
After running the notebooks, you'll find:
- `../models/best_distilbert_model.pt` - Trained model weights
- `../outputs/` - Training logs, plots, and processed data
- Performance metrics and visualizations in the notebook outputs

## � Key Features

### Data Processing
- Handles large-scale Yelp review dataset (167MB+)
- Stratified train/validation/test splitting
- Label encoding for sentiment classes
- Text tokenization with DistilBERT tokenizer

### Model Training
- Fine-tuning pre-trained DistilBERT
- Learning rate scheduling with warmup
- Early stopping based on validation loss
- Gradient accumulation for memory efficiency
- Mixed precision training support

### Evaluation & Analysis
- Comprehensive performance metrics
- Confusion matrix visualization
- Per-class precision, recall, F1-scores
- Sample predictions with confidence
- Training/validation loss curves

## 🔧 Configuration

### Hyperparameters (adjustable in notebooks)
```python
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_EPOCHS = 5
MAX_LENGTH = 512
DROPOUT_RATE = 0.3
WARMUP_STEPS = 500
```

### Hardware Requirements
- **Minimum:** 8GB RAM, CPU-only (slow training)
- **Recommended:** 16GB+ RAM, CUDA GPU with 8GB+ VRAM
- **Optimal:** 32GB+ RAM, High-end GPU (RTX 3080/4080, V100, etc.)

## � Expected Results

### Performance Benchmarks
| Dataset Split | Accuracy | F1-Score | Precision | Recall |
|--------------|----------|----------|-----------|---------|
| Validation   | ~87%     | ~85%     | ~86%      | ~84%    |
| Test         | ~85%     | ~83%     | ~84%      | ~82%    |

### Training Progress
- **Epoch 1:** Loss decreases rapidly, accuracy ~75-80%
- **Epoch 2-3:** Steady improvement, accuracy ~82-87%
- **Epoch 4-5:** Fine-tuning, marginal improvements

*Note: Results may vary based on data preprocessing and hyperparameter settings*

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size (try 8 or 4)
   - Enable gradient accumulation
   - Use smaller max sequence length (256)

2. **Slow Training**
   - Ensure CUDA is properly installed
   - Check GPU utilization with `nvidia-smi`
   - Consider using mixed precision training

3. **Poor Performance**
   - Verify data quality and preprocessing
   - Try different learning rates (1e-5, 3e-5)
   - Increase training epochs
   - Check class balance in dataset

### Memory Optimization
```python
# For limited GPU memory
torch.cuda.empty_cache()  # Clear cache between training steps
model.half()              # Use half precision
gradient_accumulation_steps = 4  # Effective batch size increase
```

## 📚 References

- [DistilBERT Documentation](https://huggingface.co/distilbert-base-uncased)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) (if used)
- [Weights & Biases](https://wandb.ai/) (for experiment tracking)

---

**Need Help?** 
- Check the main [README.md](../README.md) for project overview
- Review utility functions in [../utils/data_preprocessing_utils.ipynb](../utils/data_preprocessing_utils.ipynb)
- Open an issue in the GitHub repository for bugs or questions
