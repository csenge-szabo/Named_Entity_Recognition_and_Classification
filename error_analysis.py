import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_output_file(file_path):
    '''
    Read the output file containing predicted and gold labels.
    :param file_path: str, path to the output file
    :return: Tuple: gold_labels, predicted_labels, and lines in the file
    '''
    predicted_labels = []
    gold_labels = []
    with open(file_path, 'r') as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            splitted = line.strip('\n').split('\t')
            predicted_labels.append(splitted[-1])
            gold_labels.append(splitted[-2])
    return gold_labels, predicted_labels, lines

def analyze_misclassifications(true_label, predicted_label, gold_labels, predicted_labels, lines):
    '''
    Analyze instances where a true label was misclassified as a predicted label.
    :param true_label: str, true label to analyze misclassifications for
    :param predicted_label: str, predicted label to analyze misclassifications for
    :param gold_labels: list, true labels for all instances
    :param predicted_labels: list, predicted labels for all instances
    :param lines: list, lines from the output file
    :return: None
    '''
    misclassifications = [(i, lines[i].split('\t')[0], gold_labels[i], predicted_labels[i]) for i in range(len(gold_labels)) if
                          gold_labels[i] == true_label and predicted_labels[i] == predicted_label]

    print(f"\nInstances where '{true_label}' was misclassified as '{predicted_label}'")
    for index, token, true_label, predicted_label in misclassifications:
        print(f"Index: {index}, Token: {token}, True Label: {true_label}, Predicted Label: {predicted_label}")
        

def analyze_correct_classifications(label, gold_labels, predicted_labels, lines):
    '''
    Analyze instances where tokens were correctly classified.
    :param label: str, label to analyze correct classifications for
    :param gold_labels: list, true labels for all instances
    :param predicted_labels: list, predicted labels for all instances
    :param lines: list, lines from the output file
    :return: None
    '''
    correct_classifications = [(i, lines[i].split('\t')[0], gold_labels[i], predicted_labels[i]) for i in range(len(gold_labels)) if
                               gold_labels[i] == label and predicted_labels[i] == label]

    print(f"\nInstances where '{label}' was correctly classified")
    for index, token, true_label, predicted_label in correct_classifications:
        print(f"Index: {index}, Token: {token}, True Label: {true_label}, Predicted Label: {predicted_label}")

def main():
    svm_file = os.path.join(".", "data", "output_with_word_embeddings_and_one_hot.SVM.conll")

    gold_labels, predicted_labels, lines = read_output_file(svm_file)

    labels_for_classification_report = ["PER", "ORG", "LOC", "MISC"]
    all_labels = ["PER", "ORG", "LOC", "MISC", "O"]

    # Classification Report
    report = classification_report(gold_labels, predicted_labels, labels=labels_for_classification_report, digits=3)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    confusion_matrix_result = confusion_matrix(gold_labels, predicted_labels, labels=all_labels)
    print("Confusion Matrix:")
    print(confusion_matrix_result)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=all_labels)
    display.plot()
    plt.show()

    # Calculate misclassified instances
    misclassifications = [(i, lines[i].split('\t')[0], gold_labels[i], predicted_labels[i]) for i in range(len(gold_labels)) if gold_labels[i] != predicted_labels[i]]

    # Initialize counters
    total_misclassified = len(misclassifications)
    multi_word_ne_misclassified = 0
    single_word_ne_misclassified = 0

    # Iterate through misclassified instances and check for multi-word named entities
    for i in range(len(misclassifications) - 1):  # Avoid index out of range
        index, token, true_label, predicted_label = misclassifications[i]
        next_index, _, _, _ = misclassifications[i + 1]

        is_multi_word_ne = next_index == index + 1
        if is_multi_word_ne:
            multi_word_ne_misclassified += 1
        else:
            single_word_ne_misclassified += 1

    # Check the last instance
    last_index, _, _, _ = misclassifications[-1]
    if last_index != len(gold_labels) - 1:
        # If the last misclassification is not the last token in the sequence
        multi_word_ne_misclassified += 1
    else:
        single_word_ne_misclassified += 1

    # Display results
    print("\nMisclassification Analysis:")
    print(f"Total Misclassified: {total_misclassified}")
    print(f"Multi-Word NEs Misclassified: {multi_word_ne_misclassified}")
    print(f"Single-Word NEs Misclassified: {single_word_ne_misclassified}")

    # Analyzing 'ORG' vs. 'LOC' misclassifications:
    analyze_misclassifications("ORG", "LOC", gold_labels, predicted_labels, lines)
    analyze_misclassifications("LOC", "ORG", gold_labels, predicted_labels, lines)

    # Analyzing 'ORG' vs. 'O' misclassifications:
    analyze_misclassifications("ORG", "O", gold_labels, predicted_labels, lines)
    analyze_misclassifications("O", "ORG", gold_labels, predicted_labels, lines)

    # Analyzing correct classification examples. You can uncomment these lines to check for correct classifications.
    # analyze_correct_classifications("ORG", gold_labels, predicted_labels, lines)
    # analyze_correct_classifications("LOC", gold_labels, predicted_labels, lines)


if __name__ == "__main__":
    main()



