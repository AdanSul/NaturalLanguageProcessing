import json
from gensim.models import Word2Vec
import re
import numpy as np
import random
import time
import os
import sys
random.seed(time.time()) 
from sklearn.metrics.pairwise import cosine_similarity



def load_and_tokenize(file_path):
    original_sentences = []
    tokenized_sentences = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            text = data.get("sentence_text", "")
            # the orginal sentences
            original_sentences.append(text)
            # Tokenize using regex to extract only Hebrew words
            tokens = re.findall(r'[\u0590-\u05FF]+', text)
            if tokens:  
                tokenized_sentences.append(tokens)
    
    return original_sentences, tokenized_sentences

def load_word2vec_model(output_dir, model_name="knesset_word2vec.model"):
    model_path = os.path.join(output_dir, model_name)
    try:
        if os.path.exists(model_path):
            model = Word2Vec.load(model_path)
            return model
        else:
            return None
    except Exception:
        return None

def train_word2vec(tokenized_sentences, vector_size=50, window=5, min_count=1):
    model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model

def save_model(model, output_dir, model_name="knesset_word2vec.model"):
    output_path = os.path.join(output_dir, model_name)
    model.save(output_path)

# def test_model(model, word):
#     word_vectors = model.wv
#     if word in word_vectors:
#         print(f"Vector for '{word}':\n{word_vectors[word]}")
#     else:
#         print(f"'{word}' not found in the vocabulary.")

def tokenize_sentences(sentence):
    tokens = re.findall(r'[\u0590-\u05FF]+', sentence)
    return tokens

def find_and_save_similar_words(model, word_list, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for word in word_list:
            if word in model.wv:
                # Find the top 5 most similar words
                similar_words = model.wv.most_similar(word, topn=5)
                # the expected format
                result_line = f"{word}: " + ", ".join([f"({w}, {score:.4f})" for w, score in similar_words]) + "\n"
                file.write(result_line)
            else:
                # If the word dosnt found
                file.write(f"{word}: Word not found in vocabulary.\n")

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

def find_most_similar_sentences_random(sentence_embeddings, tokenized_sentences, original_sentences, output_file):
    # Filter sentences with at least 4 tokens
    filtered_indices = [i for i, tokens in enumerate(tokenized_sentences) if 4 <= len(tokens) <= 10 ]

    # Randomly select 10 indices from the filtered list
    selected_indices = random.sample(filtered_indices, 10)

    with open(output_file, "w", encoding="utf-8") as file:
        for idx in selected_indices:
            # Get the embedding of the current sentence
            current_embedding = sentence_embeddings[idx]

            # Compute cosine similarity with all other sentence embeddings
            similarities = cosine_similarity([current_embedding], sentence_embeddings)[0]

            # Find the most similar sentence (excluding itself)
            most_similar_idx = np.argsort(similarities)[-2]  # Second highest score (excluding itself)
            most_similar_sentence = original_sentences[most_similar_idx]

            # Write the result in the specified format
            file.write(f"{original_sentences[idx]}: most similar sentence: {most_similar_sentence}\n")

def find_most_similar_sentences(sentence_embeddings, tokenized_sentences, original_sentences, output_file, selected_indices):

    with open(output_file, "w", encoding="utf-8") as file:
        for idx in selected_indices:
            # Get the embedding of the current sentence
            current_embedding = sentence_embeddings[idx]

            # Compute cosine similarity with all other sentence embeddings
            similarities = cosine_similarity([current_embedding], sentence_embeddings)[0]

            # Find the most similar sentence (excluding itself)
            most_similar_idx = np.argsort(similarities)[-2]  # Second highest score (excluding itself)
            most_similar_sentence = original_sentences[most_similar_idx]

            # Write the result in the specified format
            file.write(f"{original_sentences[idx]}: most similar sentence: {most_similar_sentence}\n")


def replace_red_words_for_sentences(model, sentences, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for idx, (sentence, red_words) in enumerate(sentences, 1):
            new_sentence = sentence  # Start with the original sentence
            replaced_words = []

            for red_word, positives, negatives in red_words:
                try:
                    if negatives:
                        similar_word = model.wv.most_similar(positive=positives + [red_word], negative=negatives, topn=3)[0][0]
                    else:
                        similar_word = model.wv.most_similar(positive=positives + [red_word], topn=3)[0][0]

                    # Replace the word in the sentence
                    new_sentence = new_sentence.replace(red_word, similar_word)
                    replaced_words.append(f"({red_word}: {similar_word})")
                except KeyError:
                    # Handle cases where the word is not in the vocabulary
                    replaced_words.append(f"({red_word}: Word not in vocabulary)")

            # print("Original Sentence: ", sentence)
            # print("New Sentence: ", new_sentence)
            # print("replaced words: ", replaced_words)
            # Write the results to the file
            file.write(f"{idx}: {sentence}: {new_sentence}\n")
            file.write(f"replaced words: {', '.join(replaced_words)}\n")


def main(file_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    output_file_words = os.path.join(output_dir, "knesset_similar_words.txt")
    output_file_sentences = os.path.join(output_dir, "knesset_similar_sentences.txt")
    output_file_replacements = os.path.join(output_dir, "red_words_sentences.txt")

    # Load and tokenize the corpus
    original_sentences, tokenized_sentences = load_and_tokenize(file_path)
    
    # Train the Word2Vec model
    model = train_word2vec(tokenized_sentences)

    # Save the trained model
    save_model(model, output_dir)

    # Load the saved Word2Vec model
    model = load_word2vec_model(output_dir)
    
    # # Test the model 
    # sample_word = "ישראל" 
    # test_model(model, sample_word)

    # the word list
    word_list = ["ישראל", "גברת", "ממשלה", "חבר", "בוקר", "מים", "אסור", "רשות", "זכויות"]

    # Find and save similar words
    find_and_save_similar_words(model, word_list, output_file_words)

    # Calculate sentence embeddings
    sentence_embeddings = calculate_sentence_embeddings(model, tokenized_sentences)

    # find_most_similar_sentences_random(sentence_embeddings, tokenized_sentences, original_sentences, output_file_sentences)

    selected_indices = [5036, 16441, 12127, 96639, 50083, 48388, 57253, 33643, 26970, 22897]

    # Find and save most similar sentences
    find_most_similar_sentences(sentence_embeddings, tokenized_sentences, original_sentences, output_file_sentences, selected_indices)

    sentences = [
    ("בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים.",
     [
        ("דקות", ["זמן","יחידות","רגעים"], []),
        ("הדיון", ["השיחה", "הישיבה"], [])
     ]),
    ("בתור יושבת ראש הוועדה, אני מוכנה להאריך את ההסכם באותם תנאים.",
     [
        ("הוועדה", ["הממשלה","השיחה"], []),
        ("אני", ["עצמי","אישי","אחרי","הרגע"], []),
        ("ההסכם", ["חוזה"], ["סכסוך"])
     ]), 
    ("בוקר טוב, אני פותח את הישיבה.",
     [
        ("בוקר", ["יום"], ["לכולם"]),
        ("פותח", ["אקרא","אמשיך","אפעיל"], ["סוף"])
     ]),
    ("שלום, אנחנו שמחים להודיע שחברינו היקר קיבל קידום.",
     [
        ("שלום", ["אדוני","גברתי"], []),
        ("שמחים", ["מרוצים","מאושרים","מתרגשים","כולנו"], []),
        ("היקר", ["האלוף","הגדול","הטוב"], []),
        ("קידום", ["שיפור","התקדמות"], [])
     ]), 
    ("אין מניעה להמשיך לעסוק בנושא.",
     [
        ("מניעה", ["התנגדות","מגבלה"], ["הרשאה","תמיכה"])
     ])]    

    replace_red_words_for_sentences(model, sentences, output_file_replacements)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)

    # Get input folder and output file from command-line arguments
    file_path = sys.argv[1]
    output_dir = sys.argv[2]

    # file_path = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\knesset_corpus.jsonl'
    # output_dir = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\HW4'
    main(file_path, output_dir)
