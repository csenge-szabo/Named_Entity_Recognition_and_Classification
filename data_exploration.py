import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns  

def preprocess_tokens(input_filepath, output_filepath):
    """
    Take an input file, remove B- and I- label parts and write the processed file into an output file.
    :param input_filepath (str): the file to be processed
    :param output_filepath (str): the file to write with cleaned data
    :return None
    """
    column_names = ["Token", "POS-tag", "chunk-tag", "NE-label"]

    # Read data from the input file
    data = pd.read_csv(input_filepath, sep='\t', header=None, names=column_names, skip_blank_lines=False)

    # Process Named Entity labels by removing B- and I-
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
            
def plot_ne_distribution(data):
    '''
    Create a pie chart about the distribution of NE labels in the preprocessed data.
    :param data: pd.DataFrame, the preprocessed data containing NE labels
    :return: None
    '''
    # Get the distribution of NE labels
    ne_distribution = data["NE-label"].value_counts()

    # Use husl color palette for different colors in the pie chart
    colors = sns.color_palette("husl", len(ne_distribution))

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    ne_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)

    # Customize the plot
    plt.title('Distribution of NE Labels in Preprocessed Training Data')
    plt.show()
            
def get_top_tokens_for_category(category, data, n=10):
    '''
    Get the top N tokens for a given Named Entity category.
    :param category: str, the Named Entity category
    :param data: DataFrame, the data to extract tokens from
    :param n: int, the number of top tokens to retrieve
    :return: top N tokens for the given category
    '''
    category_tokens = data[data["NE-label"] == category]
    token_counts = category_tokens["Token"].value_counts().head(n)
    return token_counts

def check_capitalized_first_char(data):
    '''
    Check the percentage of Named Entity tokens with a capitalized first character.
    :param data: DataFrame, the data containing Named Entity tokens
    :return: float, percentage of Named Entity tokens with a capitalized first character
    '''
    ne_tokens = data[data["NE-label"] != "O"]
    total_ne_tokens = ne_tokens.shape[0]

    # NE tokens with a capitalized first character
    capitalized_ne_tokens = ne_tokens["Token"].apply(lambda x: x[0].isupper() if isinstance(x, str) else False).sum()

    # Percentage of NE tokens with a capitalized first character
    percentage_capitalized = (capitalized_ne_tokens / total_ne_tokens) * 100

    return percentage_capitalized

def check_pos_tag_hypothesis(data, ne_label_column="NE-label", pos_tag_column="POS-tag"):
    '''
    Check for the hypothesis about POS tags for Named Entity tokens, which claims that most NEs have the POS tag 'NNP'.
    :param data: DataFrame, the data containing Named Entity tokens
    :param ne_label_column: str, column name for Named Entity labels
    :param pos_tag_column: str, column name for POS tags
    :return: None
    '''
    ne_tokens = data[data[ne_label_column] != 'O']

    # Count the number of NE tokens with the POS tag NNP
    nnp_count = sum(ne_tokens[pos_tag_column] == 'NNP')

    # Calculate the ratio of NE tokens with the POS tag NNP
    nnp_ratio = nnp_count / len(ne_tokens)

    # Print the results
    print(f"Ratio of NE tokens with POS tag NNP: {nnp_ratio:.2%}")

def main():
    # File paths for the unprocessed data
    unprocessed_train_file_path = os.path.join(".", "data", "conll2003.train.conll")
    unprocessed_dev_file_path = os.path.join(".", "data", "conll2003.dev.conll")
    unprocessed_test_file_path = os.path.join(".", "data", "conll2003.test.conll")

    # Column names for the data
    column_names = ["Token", "POS-tag", "chunk-tag", "NE-label"]

    # Read unprocessed data
    unprocessed_train_data = pd.read_csv(unprocessed_train_file_path, sep='\t', header=None, names=column_names, skip_blank_lines=False)
    unprocessed_dev_data = pd.read_csv(unprocessed_dev_file_path, sep='\t', header=None, names=column_names, skip_blank_lines=False)
    unprocessed_test_data = pd.read_csv(unprocessed_test_file_path, sep='\t', header=None, names=column_names, skip_blank_lines=False)

    # Distribution of Named Entity classes in unprocessed data
    train_ne_distribution = unprocessed_train_data["NE-label"].value_counts()
    dev_ne_distribution = unprocessed_dev_data["NE-label"].value_counts()
    test_ne_distribution = unprocessed_test_data["NE-label"].value_counts()

    print("Distribution of NE classes in Training Data:")
    print(train_ne_distribution)
    print()

    print("Distribution of NE classes in Dev Data:")
    print(dev_ne_distribution)
    print()

    print("Distribution of NE classes in Test Data:")
    print(test_ne_distribution)
    print()

    # Preprocess and clean the data
    cleaned_train_output_path = os.path.join(".", "data", "cleaned_train_data.conll")
    preprocess_tokens(unprocessed_train_file_path, cleaned_train_output_path)

    cleaned_dev_output_path = os.path.join(".", "data", "cleaned_dev_data.conll")
    preprocess_tokens(unprocessed_dev_file_path, cleaned_dev_output_path)

    cleaned_test_output_path = os.path.join(".", "data", "cleaned_test_data.conll")
    preprocess_tokens(unprocessed_test_file_path, cleaned_test_output_path)

    # Column names for the cleaned data
    column_names = ["Token", "POS-tag", "chunk-tag", "NE-label", 'Other']

    # Read cleaned data
    train_data = pd.read_csv(cleaned_train_output_path, sep='\t', header=None, names=column_names, skip_blank_lines=False)
    dev_data = pd.read_csv(cleaned_dev_output_path, sep='\t', header=None, names=column_names, skip_blank_lines=False)
    test_data = pd.read_csv(cleaned_test_output_path, sep='\t', header=None, names=column_names, skip_blank_lines=False)

    # Filter and display distribution of Named Entity classes in preprocessed data
    selected_labels = ['O', 'MISC', 'LOC', 'PER', 'ORG']
    train_filtered = train_data[train_data["NE-label"].isin(selected_labels)]
    dev_filtered = dev_data[dev_data["NE-label"].isin(selected_labels)]
    test_filtered = test_data[test_data["NE-label"].isin(selected_labels)]

    print("Distribution of NE classes in Preprocessed Training Data:")
    print(train_filtered["NE-label"].value_counts())
    print()

    print("Distribution of NE classes in Preprocessed Dev Data:")
    print(dev_filtered["NE-label"].value_counts())
    print()

    print("Distribution of NE classes in Preprocessed Test Data:")
    print(test_filtered["NE-label"].value_counts())
    print()

    #Plot NE distribution in preprocessed training data
    plot_ne_distribution(train_data)

    # List of entity categories
    entity_categories = ["PER", "ORG", "LOC", "MISC"]

    # Extract and print the top 10 tokens for each category
    for category in entity_categories:
        top_tokens = get_top_tokens_for_category(category, train_data)
        print(f"Top 10 tokens for {category}:")
        print(top_tokens)
        print()

    # Check and print the percentage of NE tokens with a capitalized first character in the training data
    train_percentage_capitalized = check_capitalized_first_char(train_data)
    print(f"Percentage of NE tokens with a capitalized first character in Training Data: {train_percentage_capitalized:.2f}%")

    # Check and print the results of the POS tag hypothesis for the training data
    check_pos_tag_hypothesis(train_data)

if __name__ == "__main__":
    main()