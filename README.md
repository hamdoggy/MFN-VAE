# MFN-VAE
MFN-VAE

# Project Name
Multimodal Deep Learning Classification Model

## Project Overview
This project aims to classify data using a multimodal deep learning model. The project integrates data preprocessing, multimodal neural network training, loss function optimization, and hyperparameter tuning to achieve high classification accuracy.

---

## Project Structure

### Folder Structure
1. `data_pre_processing`
   - Data preprocessing module, including feature selection, data augmentation, etc.
   - Key Files:
     - `Data_pre_processing.py`
     - `load_data_sklearn.py`
     - `load_data_torch.py`
     - `smote.py`
     - `split_data(5 cross val).py`

2. `model`
   - Multimodal deep learning model definition module, supporting single- and multi-modal inputs.
   - Key Files:
     - `Multi_model.py`

3. `utils`
   - Utility module, including training, validation, and multi-task loss functions.
   - Key Files:
     - `utils.py`
     - `utils_multi_loss.py`
     - `utils_optuna.py`

4. `train`
   - Model training and optimization module, including weight training and hyperparameter optimization.
   - Key Files:
     - `train_optuna.py`
     - `train_optuna澶囦唤.py`
     - `train_weights.py`

---

## Environment Dependencies

### Required Dependencies
- Python >= 3.8
- PyTorch >= 1.9.0
- Optuna >= 2.0
- Scikit-learn >= 0.24
- Pandas >= 1.3
- Imbalanced-learn >= 0.8
- Matplotlib (optional, for visualization)

### Installation
```bash
pip install -r requirements.txt
```

---

## Data Format

### Input File Format

1. **Single-Modal Input File**:
   - **File Format**: CSV
   - **Contents**:
     - `label`: Classification label, with values `0` or `1`.
     - Remaining columns are feature data with custom column names.
   - **Example**:
     ```csv
     label,feature1,feature2,feature3,...
     0,0.12,0.34,0.56,...
     1,0.23,0.45,0.67,...
     ```

2. **Multi-Modal Input Files**:
   - **File Format**: CSV
   - **Contents**:
     - Features for different modalities are stored in separate files.
     - Each file contains a `label` column as the classification label.
     - **Example Files**:
       - `t1wi.csv`:
         ```csv
         label,feature1,feature2,feature3,...
         0,0.12,0.34,0.56,...
         1,0.23,0.45,0.67,...
         ```
       - `flair.csv`:
         ```csv
         label,feature1,feature2,feature3,...
         0,0.56,0.78,0.89,...
         1,0.65,0.87,0.98,...
         ```
       - `dwi.csv`:
         ```csv
         label,feature1,feature2,feature3,...
         0,0.34,0.67,0.78,...
         1,0.45,0.76,0.89,...
         ```

3. **Cross-Validation Files**:
   - File naming format: `crossVal_#_train.csv` and `crossVal_#_val.csv`.
   - Each fold's data is stored separately for training and validation, with the same structure as above.

### Output File Format
- **Preprocessed Data**:
  - Outputs as multiple CSV files for features and labels.
  - Example filenames: `t2wi&adc&+c_risk.csv` (features), `label.csv` (labels).

- **Model Weights**:
  - Format: `.pth`
  - Saved in the `result/` folder.

- **Training Results**:
  - Format: CSV
  - Contents: Average AUC and accuracy for different weight combinations.

---

## Usage

### 1. Data Preprocessing
Navigate to the `data_pre_processing` folder and run the following scripts:

- **Single-Modal Preprocessing**:
```bash
python Data_pre_processing.py
```

- **Multi-Modal Preprocessing**:
```bash
python Data_pre_processing.py --multi
```

### 2. Model Training
Navigate to the `train` folder and run the following scripts:

#### 2.1 Hyperparameter Optimization
Automatically tune hyperparameters using Optuna:
```bash
python train_optuna.py
```

#### 2.2 Fixed Weight Training
Train and evaluate using fixed weight combinations:
```bash
python train_weights.py
```

### 3. Model Evaluation
Use the following script in the `utils` folder to validate the model:

- **Evaluate Validation Accuracy and AUC**:
```python
from utils.utils_multi_loss import evaluate_model
accuracy, auc = evaluate_model(model, val_loader)
```

---

## Code Explanation

### Key Features
1. **Data Preprocessing**:
   - Data cleaning, standardization, feature selection (based on variance or random forest), and SMOTE augmentation.
   - Outputs preprocessed feature data and labels.

2. **Model Definition**:
   - `Encoder` and `Decoder` modules for feature extraction and reconstruction.
   - `Predictor` module for classification.
   - `MLP` model supports single- and multi-modal inputs.

3. **Loss Functions**:
   - `AutomaticWeightedLoss` and `MultiWeightedLoss` for multi-task loss weighting.

4. **Hyperparameter Optimization**:
   - Optimizes learning rate and loss weights using the Optuna framework.

---

## Output Results
1. **Best Model Weights**:
   - Saved in the `result/` folder.

2. **Training Results**:
   - Average AUC and accuracy saved in `all_result.csv`.

3. **Visualization Charts**:
   - Parameter importance, optimization history, and hyperparameter slice.

---

## Notes
1. Ensure input data format matches the requirements, especially for cross-validation paths.
2. Training time depends on the dataset size and number of hyperparameter trials.
3. Pretrained model paths need to be specified in advance.

---

## Contribution
For questions or suggestions, please contact the project maintainers.

---

## License
MIT License
