import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec
import re
import sys


def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def filter_speakers_and_labels(corpus, speakers):
    filtered_sentences = []
    labels = []
    
    for entry in corpus:
        if entry['speaker_name'] == speakers[0]:
            filtered_sentences.append(entry['sentence_text'])
            labels.append(0)
        elif entry['speaker_name'] == speakers[1]:
            filtered_sentences.append(entry['sentence_text'])
            labels.append(1)
    
    return filtered_sentences, np.array(labels)


def calculate_sentence_embeddings(model, tokenized_sentences):
    sentence_embeddings = []

    for sentence in tokenized_sentences:
        word_vectors = []

        for word in sentence:
            if word in model.wv: 
                word_vectors.append(model.wv[word])

        if word_vectors:
            # Compute the average of word vectors to get the sentence embedding
            sentence_embedding = np.mean(word_vectors, axis=0)
            sentence_embeddings.append(sentence_embedding)
        else:
            # If no words exist in the vocabulary, assign a zero vector
            sentence_embeddings.append(np.zeros(model.vector_size))

    return sentence_embeddings


def train_and_evaluate_knn(embeddings, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []
    
    for train_index, test_index in skf.split(embeddings, labels):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    # Generate classification report
    report = classification_report(y_true_all, y_pred_all, target_names=["Speaker 1", "Speaker 2"])
    print(report)


def main(file_path, model_path):

    # Loading corpus
    corpus = load_corpus(file_path)

    # the two primary speakers
    primary_speakers = ["ראובן ריבלין", "א' בורג"]

    # Filtering sentences and labels for binary classification
    filtered_sentences, labels = filter_speakers_and_labels(corpus, primary_speakers)

    # Loading Word2Vec model
    model = Word2Vec.load(model_path)

    # Tokenizing sentences
    tokenized_sentences = [re.findall(r'[\u0590-\u05FF]+', sentence) for sentence in filtered_sentences]

    # Calculating sentence embeddings
    embeddings = np.array(calculate_sentence_embeddings(model, tokenized_sentences))

    # Training and evaluating KNN model
    train_and_evaluate_knn(embeddings, labels)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    # Get input folder and output file from command-line arguments
    file_path = sys.argv[1]
    model_path = sys.argv[2]

    # file_path = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\knesset_corpus.jsonl'
    # model_path = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\HW4\knesset_word2vec.model'

    main(file_path, model_path)
