# Resume Screening Using NLP and Machine Learning

## Overview

This project implements an automated **Resume Screening and Classification System** using Natural Language Processing (NLP) and Machine Learning.

The objective is to classify resumes into their respective job categories based on the textual information present in the resumes.

The project follows an end-to-end machine learning pipeline:

**Resume Text → Data Cleaning → NLP Preprocessing → TF-IDF → Model Training → Model Evaluation → Category Prediction**

---

## Problem Statement

Recruiters often need to process a large number of resumes for different job roles. Manually categorizing resumes can be time-consuming and inefficient.

This project aims to automate the initial resume screening process by using NLP and machine learning techniques to classify resumes into predefined job categories.

The problem is treated as a **supervised multi-class classification problem**.

---

## Dataset

The dataset contains:

- 2,484 resume records
- 24 different resume categories
- Resume text and corresponding category labels

The main columns are:

| Column | Description |
|---|---|
| `ID` | Unique resume identifier |
| `Resume_str` | Text content of the resume |
| `Resume_html` | HTML representation of the resume |
| `Category` | Target job category |

Duplicate resume texts were identified and removed, resulting in **2,482 unique resumes**.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and distribution of the dataset.

The analysis includes:

- Resume category distribution
- Resume word-count analysis
- Frequently occurring words
- Corpus-level word cloud
- Identification of duplicate resumes

---

## Text Preprocessing

Resume text contains important technical terms such as Python, C++, C#, .NET, Node.js, SQL, and AWS. Therefore, preprocessing was designed to clean the text while preserving useful technical information.

The preprocessing steps include:

1. Converting text to lowercase
2. Removing HTML `<br>` tags
3. Removing URLs
4. Removing email addresses
5. Removing unwanted characters
6. Normalizing repeated dots
7. Normalizing whitespace
8. Tokenization using spaCy
9. Stopword removal
10. Punctuation removal
11. Removing selected unwanted tokens
12. Lemmatization

The cleaned text is then used for feature extraction.

---

## Feature Extraction

### TF-IDF

The cleaned resume text is converted into numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)**.

Both unigrams and bigrams are considered to capture individual words as well as short phrases.

The main configuration includes:

- `ngram_range=(1,2)`
- `max_features=15000`
- `min_df=2`
- `sublinear_tf=True`

The TF-IDF vectorizer is fitted only on the training data and then used to transform the test data to avoid data leakage.

---

## Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

A fixed `random_state=42` is used for reproducibility.

The split is stratified using the target category so that the distribution of categories is maintained in both training and testing datasets.

---

## Machine Learning Models

The following classification algorithms were trained and compared:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine
4. Random Forest
5. XGBoost

---

## Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score
- Weighted F1 Score
- Training Time

Macro-averaged metrics are important for this problem because they give equal importance to each resume category.

---

## Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **80.48%** | **82.01%** | **77.65%** | **78.15%** | **79.78%** |
| Random Forest | 73.44% | 75.44% | 68.28% | 67.33% | 71.59% |
| Linear SVM | 70.02% | 69.58% | 66.47% | 66.09% | 68.85% |
| Logistic Regression | 65.19% | 63.95% | 62.18% | 60.96% | 63.46% |
| Naive Bayes | 54.73% | 52.82% | 49.51% | 45.47% | 49.97% |

---

## Best Performing Model

Among the evaluated models, **XGBoost** achieved the best overall performance.

### XGBoost Results

- **Accuracy:** 80.48%
- **Macro Precision:** 82.01%
- **Macro Recall:** 77.65%
- **Macro F1:** 78.15%
- **Weighted F1:** 79.78%

Based on the experimental results, XGBoost was selected as the final best-performing model.

---

## Model Evaluation and Interpretation

A confusion matrix was used to analyze classification performance across the different resume categories.

XGBoost feature importance was also examined to identify important TF-IDF features contributing to the model.

This helps provide a global understanding of which terms are influential in the classification process.

---

## Project Workflow

```text
Raw Resume Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Duplicate Removal
        |
        v
Text Cleaning
        |
        v
Tokenization
        |
        v
Stopword & Punctuation Removal
        |
        v
Lemmatization
        |
        v
Train-Test Split
        |
        v
TF-IDF Feature Extraction
        |
        v
Train Multiple ML Models
        |
        v
Model Evaluation
        |
        v
Model Comparison
        |
        v
XGBoost Selected
        |
        v
Resume Category Prediction
