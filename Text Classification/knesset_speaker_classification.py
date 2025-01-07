import json
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import sys
import os

random.seed(42)
np.random.seed(42)

# Mappings for speaker name normalization
map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "הכנסת ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def update_speaker_names(corpus, alias_map):
    """Update speaker names in the corpus using an alias map."""
    for entry in corpus:
        speaker_name = entry.get('speaker_name', '')
        entry['speaker_name'] = alias_map.get(speaker_name, speaker_name)
    return corpus

def balance_classes(classification):
    """Balance classes using down-sampling."""
    class_counts = Counter([entry['class'] for entry in classification])

    min_size = min(class_counts.values())
    balanced_classification = []
    for class_label in class_counts:
        class_entries = [entry for entry in classification if entry['class'] == class_label]
        balanced_classification.extend(random.sample(class_entries, min_size))

    return balanced_classification

def classify_and_balance(corpus, primary_speakers):
    """Perform classification and balance the data."""
    binary_classification = [entry for entry in corpus if entry['speaker_name'] in primary_speakers]
    for entry in binary_classification:
        entry['class'] = primary_speakers.index(entry['speaker_name']) + 1

    multi_class_classification = []
    for entry in corpus:
        if entry['speaker_name'] in primary_speakers:
            entry['class'] = primary_speakers.index(entry['speaker_name']) + 1
        else:
            entry['class'] = 3  # "Other"
        multi_class_classification.append(entry)

    balanced_binary = balance_classes(binary_classification)
    balanced_multi_class = balance_classes(multi_class_classification)

    return balanced_binary, balanced_multi_class

def extract_tfidf_features(corpus, text_column="sentence_text"):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    texts = [entry[text_column] for entry in corpus if text_column in entry and entry[text_column].strip()]
    tfidf_matrix = vectorizer.fit_transform(texts)
    labels = np.array([entry['class'] for entry in corpus if text_column in entry and entry[text_column].strip()])
    return tfidf_matrix, labels


def extract_custom_features(corpus, text_column="sentence_text"):
    features = []
    labels = []
    protocol_types = []
    
    # Extract protocol-related metadata
    for entry in corpus:
        if text_column in entry and entry[text_column].strip():
            text = entry[text_column]
            
            # Text-derived features
            sentence_length = len(text.split())
            text_length = len(text)
            num_commas = text.count(',')
            num_periods = text.count('.')
            num_questions = text.count('?')
            num_exclamations = text.count('!')
            num_dashes = text.count('-') + text.count('–')
            
            # Metadata features
            protocol_number = float(entry.get('protocol_number', 0))
            knesset_number = float(entry.get('knesset_number', 0))
            protocol_type = entry.get('protocol_type', 'Unknown')
            protocol_types.append(protocol_type)
            
            # Append features and labels
            features.append([
                sentence_length,
                text_length,
                num_commas,
                num_periods,
                num_questions,
                num_exclamations,
                num_dashes,
                protocol_number,
                knesset_number
            ])
            labels.append(entry['class'])
    
    # Encode protocol_type as a categorical feature
    label_encoder = LabelEncoder()
    encoded_protocol_types = label_encoder.fit_transform(protocol_types)
    
    # Add encoded protocol_type to features
    for i in range(len(features)):
        features[i].append(encoded_protocol_types[i])
    
    return np.array(features), np.array(labels)



def cross_validate_and_report(X, y, classifier, classifier_name, feature_type, classification_type, cv=5):
    skf = StratifiedKFold(n_splits=cv)
    y_true_all = []
    y_pred_all = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        if isinstance(X, np.ndarray):
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # print(f"\n{classifier_name} with {feature_type} Features Classification Report:")
    target_names = [f"Class {i}" for i in np.unique(y)]
    report = classification_report(y_true_all, y_pred_all, target_names=target_names)
    # print(report)



def classify_sentences(input_file, output_file, model, vectorizer):
    try:
        # Load sentences
        with open(input_file, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file if line.strip()]

        # Transform sentences using the vectorizer
        features = vectorizer.transform(sentences)

        # Predict classes for sentences
        predictions = model.predict(features)

        # Map numeric predictions to class labels
        label_map = {1: "first", 2: "second", 3: "other"}
        predicted_labels = [label_map[pred] for pred in predictions]

        # Save predictions to output file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("\n".join(predicted_labels))

        # print(f"Classification results saved to {output_file}")

    except Exception as e:
        pass



def main(file_path, test_file, output_dir):

    output_file = os.path.join(output_dir, "classification_results.txt")

    # print("Loading corpus...")
    corpus = load_corpus(file_path)

    # print("Updating speaker names...")
    corpus = update_speaker_names(corpus, map_speakers_to_aliases)

    # print("Counting speakers...")
    speaker_counts = Counter([entry['speaker_name'] for entry in corpus if 'speaker_name' in entry])

    # print("Identifying primary speakers...")
    top_speakers = speaker_counts.most_common(2)
    primary_speakers = [speaker for speaker, _ in top_speakers]

    # print("Performing classification and balancing...")
    balanced_binary, balanced_multi_class = classify_and_balance(corpus, primary_speakers)

    # Shuffle data for better cross-validation
    balanced_binary = shuffle(balanced_binary, random_state=42)
    balanced_multi_class = shuffle(balanced_multi_class, random_state=42)

    # Binary Classification - TF-IDF
    binary_tfidf_features, binary_labels = extract_tfidf_features(balanced_binary)
    cross_validate_and_report(binary_tfidf_features, binary_labels, KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean'), "K-Nearest Neighbors", "TF-IDF", "Binary")
    cross_validate_and_report(binary_tfidf_features, binary_labels, LogisticRegression(max_iter=1000, solver='liblinear', C=10.0), "Logistic Regression", "TF-IDF", "Binary")

    # Binary Classification - Custom Features
    binary_custom_features, binary_labels = extract_custom_features(balanced_binary)
    cross_validate_and_report(binary_custom_features, binary_labels, KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean'), "K-Nearest Neighbors", "Custom", "Binary")
    cross_validate_and_report(binary_custom_features, binary_labels, LogisticRegression(max_iter=1000, solver='liblinear', C=0.1), "Logistic Regression", "Custom", "Binary")

    # Multi-Class Classification - TF-IDF
    multi_tfidf_features, multi_labels = extract_tfidf_features(balanced_multi_class)
    cross_validate_and_report(multi_tfidf_features, multi_labels, KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean'), "K-Nearest Neighbors", "TF-IDF", "Multi-Class")
    cross_validate_and_report(multi_tfidf_features, multi_labels, LogisticRegression(max_iter=1000, solver='lbfgs', C=10.0), "Logistic Regression", "TF-IDF", "Multi-Class")

    # Multi-Class Classification - Custom Features
    multi_custom_features, multi_labels = extract_custom_features(balanced_multi_class)
    cross_validate_and_report(multi_custom_features, multi_labels, KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean'), "K-Nearest Neighbors", "Custom", "Multi-Class")
    cross_validate_and_report(multi_custom_features, multi_labels, LogisticRegression(max_iter=1000, solver='lbfgs', C=0.1), "Logistic Regression", "Custom", "Multi-Class")


    # Multi-Class Classification - TF-IDF
    # print("\nTraining Multi-Class KNN Model with TF-IDF Features...")
    features, labels = extract_tfidf_features(balanced_multi_class)
    # features, labels = extract_tfidf_features(balanced_binary)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    vectorizer.fit([entry["sentence_text"] for entry in balanced_multi_class])

    # Initialize and train the KNN model
    model = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    model.fit(features, labels)

    # model = LogisticRegression(max_iter=1000, solver='liblinear', C=10.0)
    # model.fit(features, labels)


    # Classify sentences in knesset_sentences.txt
    classify_sentences(test_file, output_file, model, vectorizer)


if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     sys.exit(1)

    # # Get input folder and output file from command-line arguments
    # file_path = sys.argv[1]
    # test_file = sys.argv[2]
    # output_dir = sys.argv[3]


    file_path = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\knesset_corpus.jsonl'
    test_file = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\HW3\knesset_sentences.txt'
    output_dir = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\HW3'

    # Run the main function
    main(file_path, test_file, output_dir)