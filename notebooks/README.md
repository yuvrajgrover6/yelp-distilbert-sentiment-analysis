# Notebooks Directory

This directory contains the main Jupyter notebooks for the Yelp sentiment analysis project.

## 📓 Notebook Overview

### 1. `01_data_processing_and_training.ipynb`
**Main training notebook for the DistilBERT sentiment analysis model**

**Contents:**
- Data loading and exploration
- Text preprocessing and cleaning
- Label encoding and data splitting
- DistilBERT model configuration
- Training loop with validation
- Model evaluation and metrics
- Model saving

**Usage:**
1. Ensure dataset is in `../data/yelp_restaurants_hotels_ver2.csv`
2. Run all cells sequentially
3. Monitor training progress and validation metrics
4. Trained model will be saved to `../models/`

### 2. `02_model_inference.ipynb`
**Inference and evaluation notebook**

**Contents:**
- Loading pre-trained model
- Model evaluation on test set
- Prediction examples
- Performance visualization
- Confusion matrix analysis
- Error analysis

**Usage:**
1. Ensure trained model exists in `../models/best_distilbert_model.pt`
2. Run cells to load model and make predictions
3. Test with custom text inputs
4. Analyze model performance

## 🚀 Quick Start

### Running the Notebooks

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Run notebooks in order:**
   - First: `01_data_processing_and_training.ipynb`
   - Then: `02_model_inference.ipynb`

### Expected Runtime
- **Training notebook**: 2-3 hours (with GPU), 8-12 hours (CPU only)
- **Inference notebook**: 5-10 minutes

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 2GB+ storage space

## 🔧 Configuration

### Training Parameters
Located in `01_data_processing_and_training.ipynb`:
- `BATCH_SIZE = 16`
- `LEARNING_RATE = 2e-5`
- `NUM_EPOCHS = 3`
- `MAX_LENGTH = 512`

### Model Settings
- **Model**: `distilbert-base-uncased`
- **Number of classes**: 3 (positive, negative, neutral)
- **Dropout**: 0.3

## 📊 Expected Outputs

### Training Notebook Outputs:
- Training/validation loss plots
- Model accuracy metrics
- Saved model file (`best_distilbert_model.pt`)
- Training logs and statistics

### Inference Notebook Outputs:
- Test set performance metrics
- Confusion matrix visualization
- Sample predictions with confidence scores
- Model evaluation summary

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA out of memory**
   - Reduce `BATCH_SIZE` from 16 to 8 or 4
   - Enable gradient accumulation

2. **Dataset not found**
   - Ensure `yelp_restaurants_hotels_ver2.csv` is in `../data/` directory
   - Check file path in notebook

3. **Model loading errors**
   - Verify model file exists in `../models/`
   - Check file permissions

4. **Slow training**
   - Ensure GPU is being used (`torch.cuda.is_available()`)
   - Consider using smaller dataset for testing

## 💡 Tips

- **Memory optimization**: Close other applications when training
- **Monitoring**: Use `nvidia-smi` to monitor GPU usage
- **Experimentation**: Try different hyperparameters in separate copies
- **Backup**: Save intermediate results during long training runs

## 📝 Notes

- Notebooks are designed to run independently
- All file paths are relative to the notebook location
- Progress bars show training/inference progress
- Results are automatically saved to `../outputs/` directory

For questions or issues, please refer to the main project README or create an issue in the repository.
