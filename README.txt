# Named Entity Recognition and Classification

## Author
Csenge Szabó

This repository contains the project work for the Machine Learning Master's course in Computational Linguistics taught at VU Amsterdam, focusing on Named Entity Recognition and Classification (NERC).

## Prerequisites
Before you begin, ensure you complete the following steps:
1. **Download Google word embeddings**:
   - Download from [Google Word2Vec](https://code.google.com/archive/p/word2vec/).
   - Place the file `GoogleNews-vectors-negative300.bin.gz` in the `model` subdirectory.
2. **Install Required Libraries**:
   - Ensure Python modules listed in `requirements.txt` are installed by running:


## Folder Structure
- `NERC_Report_Csenge_Szabo.pdf`: Detailed project report.
- `requirements.txt`: Lists required Python libraries.
- `data`: Contains training, development, and test datasets.
- `model`: Directory to store the Google Word Embeddings file.

## Python Scripts
### 1. `data_exploration.py`
- **Description**: Preprocesses data, extracts information, and creates plots about NE class distribution. Analyzes data based on capitalization patterns and most frequent POS-tags.
    
### 2. `models_with_one_hot_encoding.py`
- **Description**: Preprocesses files, trains Logistic Regression, Multinomial Naive Bayes, and Support Vector Machines classifiers with one-hot encoded features from the training data, tests and evaluates them using the test data. 

### 3. `feature_ablation.py`
- **Description**: Preprocesses files, trains the Logistic Regression, Multinomial Naive Bayes, Support Vector Machines classifiers with one-hot encoded features from the training data, tests and evaluates them using the test data. Evaluation is done in separate steps for each feature, starting with the token feature and adding one more feature at a time.

### 4. `models_with_word_embeddings.py`
- **Description**: Preprocesses files, trains Logistic Regression and Support Vector Machines classifiers with word embeddings from the training data, tests, and evaluates them using the test data.

### 5. `models_with_combined_features.py`
- **Description**: Preprocesses files, trains Logistic Regression and Support Vector Machines classifiers with the combination of one-hot encoded features and word embeddings from the training data, tests, and evaluates them using the test data.

### 6. `finetuning_SVM_with_combined_features.py`
- **Description**: Preprocesses files, fine-tunes the hyper-parameters of the Support Vector Machines classifier with combined features (one-hot encoded features and word embeddings). It splits the data into training and testing sets, performs hyperparameter tuning using Randomized Search and cross-validation, trains the SVM classifier with the best hyperparameters, and evaluates using the F1 score.

### 7. `error_analysis.py`
- **Description**: Reads the input CoNLL file (output of the best-performing SVM classifier created by 'models_with_combined features.py'). It analyzes misclassifications and correct classifications for certain NE classes, and also analyzes the ratio of single-token vs multi-token NE misclassifications.

REMARKS:
Throughout the semester I collaborated with Murat Ertaş, we developed some parts of code together, especially the code for extracting and combining word embeddings. Discussions with Christina Karavida inspired me to explore the idea of advanced features, particularly 'word shape'.

