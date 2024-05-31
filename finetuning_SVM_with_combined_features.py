from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import gensim
import csv

def preprocess_tokens(input_filepath, output_filepath):
    """
    Take an input file, remove B- and I- label parts and write the processed file into an output file.
    :param input_filepath (str): the file to be processed
    :param output_filepath (str): the file to write with cleaned data
    :return None
    """
    column_names = ["Token", "POS-tag", "chunk-tag", "NE-label"]

    #Read data from the input file
    data = pd.read_csv(input_filepath, sep='\t', header=None, names=column_names, skip_blank_lines=False)

    #Process Named Entity labels by removing B- and I-
    ne_data = data["NE-label"]
    token_categories = []
    
    for cat in ne_data:
        if pd.notna(cat) and cat != 'O':
            if cat.startswith('B-') or cat.startswith('I-'):
                cat = cat[2:]
        token_categories.append(cat)

    data["NE-label"] = token_categories
  
    # Save the preprocessed data to the output file in CoNLL format
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for i, row in data.iterrows():
            outfile.write('\t'.join(map(str, row)) + '\n')

def extract_embeddings_as_features_and_gold(conllfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings of current, next and previous token within a sentence.
    :param conllfile: str, path to conll file
    :param word_embedding_model: a pretrained word embedding model (gensim.models.keyedvectors.Word2VecKeyedVectors)
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t', quotechar='|')

    current_sentence_tokens = []

    for row in csvreader:
        # Check for cases where the row is not empty
        if row:
            current_token = row[0]
            if current_token in word_embedding_model:
                current_vector = word_embedding_model[current_token]
            else:
                current_vector = [0] * 300

            current_sentence_tokens.append((current_token, current_vector, row[-1]))

    # Process the entire sequence of tokens for the sentence
    for i in range(len(current_sentence_tokens)):
        current_token, current_vector, label = current_sentence_tokens[i]

        # Use a zero vector for the previous token if it's the first token in the sentence
        prev_vector = current_sentence_tokens[i - 1][1] if i > 0 else [0] * 300

        # Use a zero vector for the next token if it's the last token in the sentence
        next_vector = current_sentence_tokens[i + 1][1] if i < len(current_sentence_tokens) - 1 else [0] * 300

        # Combine current, previous, and next token's word embeddings
        combined_vector = np.concatenate((prev_vector, current_vector, next_vector), axis=-1)

        features.append(combined_vector)
        labels.append(label)

    return features, labels

def word_shape(token):
    """
    Generate a word shape representation for a given token (e.g. 'Ccco' for 'Ltd.').
    :param token: str, input token
    :return: str, word shape representation
    """
    return ''.join(['d' if char.isdigit() 
                    else 'c' if char.islower() 
                    else 'C' if char.isupper() 
                    else 'o' for char in token])

def extract_features_and_labels(trainingfile):
    """
    Extract features and labels from a training file.
    :param trainingfile: str, path to the training file
    :return: Tuple of lists: extracted features and labels
    """
    data = []
    targets = []

    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            
            if len(components) > 1:
                token = components[0]
                pos_tag = components[1]
                shape = word_shape(token)

                feature_dict = {
                    'POS-tag': pos_tag,
                    'word_shape': shape}
        
                data.append(feature_dict)
                targets.append(components[-1])
                
    return data, targets


def combine_features(features_vectorized, word_embedding_features):
    """
    Combine vectorized features and word embedding features to create a single feature matrix
    :param features_vectorized: vectorized features
    :param word_embedding_features: list of concatenated word embedding features
    :return: numpy array of combined features
    """
    features_vec_array = features_vectorized.toarray()

    combined_features = np.concatenate((word_embedding_features, features_vec_array), axis=-1)
    return combined_features

# Code was adapted from the sklearn documentation: https://scikit-learn.org/stable/modules/grid_search.html
def main(argv=None):
    print("Start of the script.")
    if argv is None:
        argv = sys.argv
    
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
            
    # Load  data and extract features and embeddings
    language_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(".", "model", "GoogleNews-vectors-negative300.bin"), binary=True)
    training_features, gold_labels = extract_features_and_labels(trainingfile)
    word_embedding_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)

    # Combine features
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(training_features)
    combined_features = combine_features(features_vectorized, word_embedding_features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, gold_labels, test_size=0.2, random_state=42)

    # Define the parameter distributions for random search
    param_dist = {
        'C': (0.001, 0.1, 1, 10),
        'gamma': ['scale'],
        'kernel': ['linear', 'rbf']}

    # Initialize SVM
    svm = SVC()

    # Create Randomized Search
    random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist,
        n_iter=8,
        scoring='f1_macro',
        cv=5,
        verbose=2)

    # Fit the model on the training data
    random_search.fit(X_train, y_train)

    print("Best Hyperparameters:", random_search.best_params_)

    # Get the best SVM model from the randomized search
    best_svm_model_random = random_search.best_estimator_
    print(best_svm_model_random)

    # Evaluate the model on the test set using F1 score
    y_pred = best_svm_model_random.predict(X_test)
    test_f1_score = f1_score(y_test, y_pred, average='macro')
    print("Test Set F1 Score:", test_f1_score)
    
if __name__ == '__main__':
    # Preprocessing the train and dev set
    train_input_path = os.path.join(".", "data", "conll2003.train.conll")
    cleaned_train_output_path = os.path.join(".", "data", "cleaned_train_data.conll")
    preprocess_tokens(train_input_path, cleaned_train_output_path)

    dev_input_path = os.path.join(".", "data", "conll2003.dev.conll")
    cleaned_dev_output_path = os.path.join(".", "data", "cleaned_dev_data.conll")
    preprocess_tokens(dev_input_path, cleaned_dev_output_path)

    # Set the paths for training, input, and output files
    trainingfile = os.path.join(".", "data", "cleaned_train_data.conll")
    inputfile = os.path.join(".", "data", "cleaned_dev_data.conll")
    outputfile = os.path.join(".", "data", "output_with_combined_features_finetuned.conll")

    # Pass the arguments to the main function
    args = ['my_python', trainingfile, inputfile, outputfile]
    main(args)