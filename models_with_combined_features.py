from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

        # Use a zero vector for the previous token if it is the first token in the sentence
        prev_vector = current_sentence_tokens[i - 1][1] if i > 0 else [0] * 300

        # Use a zero vector for the next token if it is the last token in the sentence
        next_vector = current_sentence_tokens[i + 1][1] if i < len(current_sentence_tokens) - 1 else [0] * 300

        # Combine current, previous, and next token is word embeddings
        combined_vector = np.concatenate((prev_vector, current_vector, next_vector), axis=-1)

        features.append(combined_vector)
        labels.append(label)

    return features, labels

def extract_embeddings_as_features(conllfile,word_embedding_model):
    '''
    Function that extracts features using word embeddings of current, next and previous token within a sentence.
    :param conllfile: str, path to conll file
    :param word_embedding_model: a pretrained word embedding model (gensim.models.keyedvectors.Word2VecKeyedVectors)
    :return features: list of vector representation of tokens
    '''
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

        # Use a zero vector for the previous token if it is the first token in the sentence
        prev_vector = current_sentence_tokens[i - 1][1] if i > 0 else [0] * 300

        # Use a zero vector for the next token if it is the last token in the sentence
        next_vector = current_sentence_tokens[i + 1][1] if i < len(current_sentence_tokens) - 1 else [0] * 300

        # Combine current, previous, and next token's word embeddings
        combined_vector = np.concatenate((prev_vector, current_vector, next_vector), axis=-1)

        features.append(combined_vector)

    return features

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


def extract_features(inputfile):
    """
    Extract features from an input file.
    :param inputfile: str, path to the input file
    :return: list, extracted features
    """
    data = []

    with open(inputfile, 'r', encoding='utf8') as infile:
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

    return data

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

def create_classifier(train_features, train_targets, modelname, word_embedding_features):
    """
    Create a classifier based on the specified modelname.
    :param train_features: list, training features
    :param train_targets: list, training targets
    :param modelname: str, name of the model ('logreg', 'SVM')
    :param word_embedding_features: list, concatenated word embedding features
    :return: Tuple of trained model and vectorizer
    """
    print('Creating classifier...')
    vec = DictVectorizer()
                
    if modelname == 'logreg':
        model = LogisticRegression(max_iter=10000)
    elif modelname == 'SVM':
        model = SVC(kernel='rbf', C=10, gamma='scale')
    else:
        raise ValueError(f"Unsupported modelname: {modelname}")
        
    features_vectorized = vec.fit_transform(train_features)
    combined_features = combine_features(features_vectorized, word_embedding_features)
    model = model.fit(combined_features, train_targets)
   
    return model, vec
    
def classify_data(model, vec, inputdata, outputfile, word_embedding_model):
    """
    Classify new data using a trained model and word embedding model.
    :param model: trained classifier model (object)
    :param vec: vectorizer (object)
    :param inputdata: str, path to the input data
    :param outputfile: str, path to the output file
    :param word_embedding_model: a pretrained word embedding model (gensim.models.keyedvectors.Word2VecKeyedVectors)
    :return: None
    """
    print('Classify new data...')
    features = extract_features(inputdata)
    features = vec.transform(features)
    word_embedding_features = extract_embeddings_as_features(inputdata, word_embedding_model)
    combined_features = combine_features(features, word_embedding_features)
    predictions = model.predict(combined_features)

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
    print(f"\nResults for the {modelname} model with word embeddings + one-hot encoded features:")
    predicted_labels = []
    gold_labels = []
    
    # Extracting predicted labels and gold labels
    with open(outputfile, 'r') as output:
        lines = output.readlines()
        for line in lines:
            splitted = line.strip('\n').split('\t')
            predicted_labels.append(splitted[-1])
            gold_labels.append(splitted[-2])

    # Specify the labels of interest
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
    
    language_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(".", "model", "GoogleNews-vectors-negative300.bin"), binary=True)
    
    training_features, gold_labels = extract_features_and_labels(trainingfile)
    word_embedding_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)
    
    for modelname in ['logreg', 'SVM']:
        ml_model, vec = create_classifier(training_features, gold_labels, modelname, word_embedding_features)
        classify_data(ml_model, vec, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'), language_model)
        evaluate_model(outputfile.replace('.conll','.' + modelname + '.conll'), ml_model)
    
if __name__ == '__main__':
    # Preprocessing the train and test set
    train_input_path = os.path.join(".", "data", "conll2003.train.conll")
    cleaned_train_output_path = os.path.join(".", "data", "cleaned_train_data.conll")
    preprocess_tokens(train_input_path, cleaned_train_output_path)

    test_input_path = os.path.join(".", "data", "conll2003.test.conll")
    cleaned_test_output_path = os.path.join(".", "data", "cleaned_test_data.conll")
    preprocess_tokens(test_input_path, cleaned_test_output_path)

    # Set the paths for training, input, and output files
    trainingfile = os.path.join(".", "data", "cleaned_train_data.conll")
    inputfile = os.path.join(".", "data", "cleaned_test_data.conll")
    outputfile = os.path.join(".", "data", "output_with_combined_features.conll")

    # Pass the arguments to the main function
    args = ['my_python', trainingfile, inputfile, outputfile]
    main(args)