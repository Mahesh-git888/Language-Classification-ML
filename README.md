
# **Language Classification using Machine Learning**

This project is focused on classifying languages based on text data using machine learning techniques. It uses a Random Forest model trained on features extracted from text, such as word length, sentence length, vowel count, stopword count, and script-based features. The datasets include multilingual text samples for training and testing.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Datasets](#datasets)
4. [Requirements](#requirements)
5. [Usage Instructions](#usage-instructions)
6. [Project Structure](#project-structure)

---

## **Project Overview**
The goal of this project is to classify text into different Indian languages using extracted features and a machine learning model. It demonstrates:
- Feature extraction techniques for text-based data.
- The application of Random Forest for multi-class classification.
- Handling multilingual datasets efficiently.

---

## **Features**
- **Feature Extraction**:
  - Average word length.
  - Sentence length.
  - Vowel count.
  - Stopword count.
  - Script-based features.
- **Model Training**:
  - Random Forest model for classification.
  - One-hot encoding for categorical features.
- **Data Visualization**:
  - Confusion matrix for model performance.
  - Metrics like accuracy and F1 score.

---

## **Datasets**
1. **Main Dataset**: `languages1.csv`  
   - Contains multilingual text data.
   - Used for feature extraction and analysis.

2. **Training Dataset**: `training_dataset.csv`  
   - Subset of the main dataset for training.

3. **Test Dataset**: `test_dataset.csv`  
   - Subset of the main dataset for testing.

4. **Processed Data**: `processed_data_with_features.csv`  
   - Contains the dataset after feature extraction.

---

## **Requirements**
- Python 3.8 or higher
- Libraries:
  - `numpy`
  - `pandas`
  - `sklearn`
  - `matplotlib`
  - `nltk`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Usage Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/language-classification-ml.git
   cd language-classification-ml
   ```

2. Run the notebooks:
   - **Training Notebook**: `final_TrainPR.ipynb`
   - **Testing Notebook**: `final_testPR.ipynb`

3. Input datasets (`languages1.csv`, `training_dataset.csv`, `test_dataset.csv`) should be placed in the `data/` directory.

4. Outputs and results (e.g., confusion matrix, metrics) will be generated during execution.

---

## **Project Structure**
```
language-classification-ml/
│
├── data/
│   ├── languages1.csv
│   ├── training_dataset.csv
│   ├── test_dataset.csv
│   └── processed_data_with_features.csv
│
├── notebooks/
│   ├── final_TrainPR.ipynb
│   ├── final_testPR.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   └── metrics.txt
│
├── README.md
└── requirements.txt
```

