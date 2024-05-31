from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import os

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
    print('Extracting features...')
    data = []
    targets = []
    
    prev_token = ''

    with open(trainingfile, 'r', encoding='utf8') as infile:
        lines = infile.readlines()  # read all the lines of the file into a list
        for current_line, next_line in zip(lines, lines[1:] + ['']):
            current_components = current_line.rstrip('\n').split()
            next_components = next_line.rstrip('\n').split()

            if len(current_components) > 1:
                token = current_components[0]
                pos_tag = current_components[1]
                shape = word_shape(token)
            
                feature_dict = {
                    'token': token,
                    'word_shape': shape,
                    'prev_token': prev_token,
                    'next_token': next_components[0] if len(next_components) > 0 else 'END',
                    'POS-tag': pos_tag}

                data.append(feature_dict)

                prev_token = token

                targets.append(current_components[-1])
    return data, targets

def extract_features(inputfile):
    """
    Extract features from an input file.
    :param inputfile: str, path to the input file
    :return: list, extracted features
    """
    data = []
    prev_token = ''

    with open(inputfile, 'r', encoding='utf8') as infile:
        lines = infile.readlines()  # read all the lines of the file into a list
        for current_line, next_line in zip(lines, lines[1:] + ['']):
            current_components = current_line.rstrip('\n').split()
            next_components = next_line.rstrip('\n').split()

            if len(current_components) > 1:
                token = current_components[0]
                pos_tag = current_components[1]
                shape = word_shape(token)
            
                feature_dict = {
                    'token': token,
                    'word_shape': shape,
                    'prev_token': prev_token,
                    'next_token': next_components[0] if len(next_components) > 0 else 'END',
                    'POS-tag': pos_tag}

                data.append(feature_dict)

                prev_token = token

    return data

def create_classifier(train_features, train_targets, modelname):
    """
    Create a classifier based on the specified modelname.
    :param train_features: list, training features
    :param train_targets: list, training targets
    :param modelname: str, name of the model ('logreg', 'NB', 'SVM')
    :return: Tuple of trained model and vectorizer
    """
    print('Creating classifier...')
    vec = DictVectorizer()
    if modelname == 'logreg':
        model = LogisticRegression(max_iter=10000)
    
    elif modelname == 'NB':
        model = MultinomialNB()
        
    elif modelname == 'SVM':
        model = SVC(kernel = 'linear', C=1.0)

    features_vectorized = vec.fit_transform(train_features)
    model = model.fit(features_vectorized, train_targets)
    
    return model, vec

def classify_data(model, vec, inputdata, outputfile):
    """
    Classify new data using a trained model and vectorizer.
    :param model: trained classifier model (object)
    :param vec: vectorizer (object)
    :param inputdata: str, path to the input data
    :param outputfile: str, path to the output file
    :return: None
    """
    print('Classify new data...')
    
    features = extract_features(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

def evaluate_model(outputfile, modelname):
    """
    Evaluate the performance of a model using classification report and confusion matrix.
    :param outputfile: str, path to the output file
    :param modelname: str, name of the model
    :return: None
    """
    print(f"\nResults for the {modelname} model with five one-hot encoded features:")
    predicted_labels = []
    gold_labels = []
    
    # Extracting predicted labels and gold labels
    with open(outputfile, 'r') as output:
        lines = output.readlines()
        for line in lines:
            splitted = line.strip('\n').split('\t')
            predicted_labels.append(splitted[-1])
            gold_labels.append(splitted[-2])

    # Specify the labels of interest (exclude 'O' class)
    labels_for_classification_report = ["PER", "ORG", "LOC", "MISC"]
    all_labels = ["PER", "ORG", "LOC", "MISC", "O"]

    report = classification_report(gold_labels, predicted_labels, labels=labels_for_classification_report, digits=3)
    print("Classification Report:")
    print(report)

    confusion_matrix_result = confusion_matrix(gold_labels, predicted_labels, labels=all_labels)
    print("Confusion Matrix:")
    print(confusion_matrix_result)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=all_labels)
    display.plot()
    plt.show()
    
def main(argv=None):
    print("Start of the script.")
    if argv is None:
        argv = sys.argv
    
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    
    training_features, gold_labels = extract_features_and_labels(trainingfile)

    for modelname in ['logreg', 'NB', 'SVM']:
        ml_model, vec = create_classifier(training_features, gold_labels, modelname)
        classify_data(ml_model, vec, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))
        evaluate_model(outputfile.replace('.conll','.' + modelname + '.conll'), ml_model)
    
if __name__ == '__main__':
    # Preprocessing the train set
    train_input_path = os.path.join(".", "data", "conll2003.train.conll")
    cleaned_train_output_path = os.path.join(".", "data", "cleaned_train_data.conll")
    preprocess_tokens(train_input_path, cleaned_train_output_path)

    # Preprocessing the test set
    test_input_path = os.path.join(".", "data", "conll2003.test.conll")
    cleaned_test_output_path = os.path.join(".", "data", "cleaned_test_data.conll")
    preprocess_tokens(test_input_path, cleaned_test_output_path)

    # Set the paths for training, input, and output files
    trainingfile = os.path.join(".", "data", "cleaned_train_data.conll")
    inputfile = os.path.join(".", "data", "cleaned_test_data.conll")
    outputfile = os.path.join(".", "data", "output_one_hot_features.conll")

    # Pass the arguments to the main function
    args = ['my_python', trainingfile, inputfile, outputfile]
    main(args)