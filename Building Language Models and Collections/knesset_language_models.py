import json
from collections import Counter
import math
import os
import random
import sys


class Trigram_LM:
    def __init__(self, sentences):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()
        self.total_tokens = 0
        self.build_model(sentences)

    def build_model(self, sentences):
        for sentence in sentences:
            tokens = sentence
            # Update total token count and vocabulary set
            self.total_tokens += len(tokens)
            self.vocab.update(tokens)
            # Update counts
            self.unigrams.update(tokens)
            self.bigrams.update(zip(tokens[:-1], tokens[1:]))
            self.trigrams.update(zip(tokens[:-2], tokens[1:-1], tokens[2:]))

    def calculate_prob_of_sentence(self, sentence):
      
        tokens = sentence.split()
        number_words = len(tokens)
        log_prob = 0.0  # Initialize log probability

        # weights for linear interpolation
        λ1 = 0.1
        λ2 = 0.3
        λ3 = 0.6

        vocab_size = len(self.vocab)

        for i in range(number_words):
            # Unigram probability with Laplace smoothing
            unigram_prob = (self.unigrams.get(tokens[i], 0) + 1) / (self.total_tokens + vocab_size)

            # Bigram probability with Laplace smoothing
            if i > 0:
                bigram = (tokens[i - 1], tokens[i])
                bigram_prob = (self.bigrams.get(bigram, 0) + 1) / (self.unigrams.get(tokens[i - 1], 0) + vocab_size)
            else:
                bigram_prob = 0  # No bigram context for the first token

            # Trigram probability with Laplace smoothing
            if i > 1:
                trigram = (tokens[i - 2], tokens[i - 1], tokens[i])
                trigram_prob = (self.trigrams.get(trigram, 0) + 1) / (self.bigrams.get((tokens[i - 2], tokens[i - 1]), 0) + vocab_size)
            else:
                trigram_prob = 0  # No trigram context for the first two tokens

            # Combine probabilities using linear interpolation
            interpolated_prob = λ1 * unigram_prob + λ2 * bigram_prob + λ3 * trigram_prob

            # Handle zero probabilities
            if interpolated_prob == 0:
                interpolated_prob = 1e-12  # Avoid log(0)

            # Add log probability
            log_prob += math.log2(interpolated_prob)

        return log_prob
    
    def generate_next_token(self, context):
        max_prob =  float('-inf') 
        best_token = None  

        w_k_2, w_k_1 = context  
        vocab_size = len(self.vocab)  

        for w_k in self.vocab:
            if w_k in {"<s_0>", "<s_1>"}:  
                continue

            if w_k_2 is None and w_k_1 is None:
                unigram_count = self.unigrams.get(w_k, 0)
                prob = (unigram_count + 1) / (self.total_tokens + vocab_size)

            else:
                trigram_count = self.trigrams.get((w_k_2, w_k_1, w_k), 0)
                bigram_count = self.bigrams.get((w_k_2, w_k_1), 0)
                prob = (trigram_count + 1) / (bigram_count + vocab_size)

            if prob > max_prob:
                max_prob = prob
                best_token = w_k

        return best_token if best_token else "<UNK>", max_prob

    def calculate_tfidf(self, sentences, n, t):
        total_documents = len(sentences)
        ngram_document_frequency = Counter()

        # Calculate TF and Document Frequencies
        ngram_tf = {}
        for sentence in sentences:
            words = sentence
            ngrams_in_sentence = [
                tuple(words[i:i + n]) for i in range(len(words) - n + 1) if all(word not in ['<s_0>', '<s_1>'] for word in words[i:i + n])]

            total_ngrams = len(ngrams_in_sentence)

            if total_ngrams == 0:
                continue

            # Count TF for n-grams in this document
            ngram_counts = Counter(ngrams_in_sentence)
            for ngram, count in ngram_counts.items():
                tf = count / total_ngrams
                ngram_tf.setdefault(ngram, []).append(tf)  # Store TF values
                ngram_document_frequency[ngram] += 1  # Document Frequency

        # Calculate TF-IDF Scores
        tfidf_scores = {}
        for ngram, tf_list in ngram_tf.items():
            if ngram_document_frequency[ngram] >= t:  # Check threshold
                idf = math.log2(total_documents / ngram_document_frequency[ngram])  # IDF calculation
                for tf in tf_list:  # Multiply TF and IDF
                    tfidf_scores[ngram] = tfidf_scores.get(ngram, 0) + (tf * idf)

        return tfidf_scores

    def get_k_n_t_collocations(self, k, n, t, corpus, score_type):

        # Extract n-grams from the corpus
        ngrams = []
        for sentence in corpus:
            ngrams_in_sentence = [
                tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)
                 if all(word not in ['<s_0>', '<s_1>'] for word in sentence[i:i + n])]
        
            ngrams.extend(ngrams_in_sentence)

        # Count occurrences of n-grams
        ngram_counts = Counter(ngrams)

        # frequency threshold `t`
        filtered_ngrams = {ngram: count for ngram, count in ngram_counts.items() if count >= t}

        # Frequency-based ranking
        if score_type == "Frequency":
            most_common_ngrams = sorted(filtered_ngrams.items(), key=lambda x: x[1], reverse=True)[:k]
            return [' '.join(collocation[0]) for collocation in most_common_ngrams]

        # TF-IDF-based ranking
        elif score_type == "TF-IDF":
            tfidf_scores = self.calculate_tfidf(corpus, n, t)
            sorted_tfidf_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            # return [' '.join(ngram[0]) for ngram, _ in sorted_tfidf_scores]
            return [' '.join(ngram) for ngram, _ in sorted_tfidf_scores]
        else:
            raise ValueError("Unsupported metric")

    def calculate_perplexity(self, original_sentence, masked_sentence, restored_sentence):
    
        original_tokens = original_sentence.split()
        masked_tokens = masked_sentence.split()
        restored_tokens = restored_sentence.split()

        log_prob_sum = 0.0
        masked_count = 0  # Count only the masked tokens

        vocab_size = len(self.vocab)

        for i, token in enumerate(masked_tokens):
            if token == "[*]":  # Focus only on masked positions
                masked_count += 1
                restored_token = restored_tokens[i]

                # Calculate interpolated probability
                # Unigram
                unigram_prob = (self.unigrams.get(restored_token, 0) + 1) / (self.total_tokens + vocab_size)

                # Bigram
                if i > 0:
                    bigram = (restored_tokens[i - 1], restored_token)
                    bigram_prob = (self.bigrams.get(bigram, 0) + 1) / (self.unigrams.get(restored_tokens[i - 1], 0) + vocab_size)
                else:
                    bigram_prob = 0

                # Trigram
                if i > 1:
                    trigram = (restored_tokens[i - 2], restored_tokens[i - 1], restored_token)
                    trigram_prob = (self.trigrams.get(trigram, 0) + 1) / (self.bigrams.get((restored_tokens[i - 2], restored_tokens[i - 1]), 0) + vocab_size)
                else:
                    trigram_prob = 0

                # Linear interpolation
                lambda1, lambda2, lambda3 = 0.1, 0.3, 0.6
                interpolated_prob = lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob

                if interpolated_prob == 0:
                    interpolated_prob = 1e-12  # Avoid log(0)

                log_prob_sum += math.log2(interpolated_prob)

        if masked_count == 0:
            return float('inf')  # Avoid division by zero

        # Calculate perplexity for masked tokens
        avg_log_prob = log_prob_sum / masked_count
        perplexity = 2 ** (-avg_log_prob)
        return perplexity

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    if 'protocol_type' in entry and 'sentence_text' in entry:
                        data.append(entry)
                    else:
                        print(f"Skipping invalid entry: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Invalid JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    return data

def get_sentences_by_type(data, protocol_type):
    if not data:
        print("Warning: No data available to filter sentences.")
        return []
    sentences = [entry['sentence_text'] for entry in data if entry.get('protocol_type') == protocol_type]
    if not sentences:
        print(f"Warning: No sentences found for type '{protocol_type}'.")
    return sentences

def add_start_tokens(corpus):
    start_tokens = ["<s_0>", "<s_1>"] 
    processed_corpus = [start_tokens + sentence.split() for sentence in corpus]
    return processed_corpus

def mask_tokens_in_sentences(sentences, x):
    if not (0 < x <= 1):
        raise ValueError("x must be a percentage between 0 and 1.")

    masked_sentences = []

    for sentence in sentences:
        # Determine the number of tokens to mask based on the percentage
        raw_num_tokens_to_mask = len(sentence) * x

        # Handle rounding logic
        if raw_num_tokens_to_mask - int(raw_num_tokens_to_mask) >= 0.5:
            num_tokens_to_mask = math.ceil(raw_num_tokens_to_mask)  # Round up
        else:
            num_tokens_to_mask = math.floor(raw_num_tokens_to_mask)  # Round down

        # Ensure at least one token is masked
        num_tokens_to_mask = max(1, num_tokens_to_mask)

        # Randomly select `num_tokens_to_mask` indices in the sentence
        mask_indices = random.sample(range(len(sentence)), num_tokens_to_mask)

        # Create a masked copy of the sentence
        masked_sentence = sentence.copy()
        for idx in mask_indices:
            masked_sentence[idx] = '[*]'

        # Add the masked sentence to the list
        masked_sentences.append(masked_sentence)

    return masked_sentences

def sample_and_save_sentences(sentences, x, num_samples, original_file, masked_file):
    # Filter sentences to include only those with at least 5 tokens
    filtered_sentences = [sentence for sentence in sentences if len(sentence) >= 5]

    # Randomly sample `num_samples` sentences
    sampled_sentences = random.sample(filtered_sentences, num_samples)

    # Mask `x%` of tokens in the sampled sentences
    masked_sentences = mask_tokens_in_sentences(sampled_sentences, x)

    # Save original sampled sentences to file
    with open(original_file, 'w', encoding='utf-8') as f:
        for sentence in sampled_sentences:
            f.write(' '.join(sentence) + '\n')

    # Save masked sampled sentences to file
    with open(masked_file, 'w', encoding='utf-8') as f:
        for sentence in masked_sentences:
            f.write(' '.join(sentence) + '\n')

def restore_and_evaluate_sentences(committee_model, plenary_model, original_file, masked_file, output_file):
    # Load original and masked sentences
    with open(original_file, 'r', encoding='utf-8') as f:
        original_sentences = [line.strip() for line in f]
    with open(masked_file, 'r', encoding='utf-8') as f:
        masked_sentences = [line.strip().split() for line in f]

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, (original, masked) in enumerate(zip(original_sentences, masked_sentences)):
            f_out.write(f"original_sentence: {original}\n")
            f_out.write(f"masked_sentence: {' '.join(masked)}\n")

            # Restore missing tokens using the plenary model
            restored_sentence = masked.copy()
            generated_tokens = []  # Track only the generated tokens
            for idx, token in enumerate(masked):
                if token == '[*]':
                    context = (
                        restored_sentence[idx - 2] if idx >= 2 else '<s_0>',
                        restored_sentence[idx - 1] if idx >= 1 else '<s_1>'
                    )
                    predicted_token, _ = plenary_model.generate_next_token(context)
                    restored_sentence[idx] = predicted_token
                    generated_tokens.append(predicted_token)  # Save only generated tokens

            restored_sentence_str = ' '.join(restored_sentence)
            f_out.write(f"plenary_sentence: {restored_sentence_str}\n")
            f_out.write(f"plenary_tokens: {','.join(generated_tokens)}\n")

            # Calculate log probabilities
            log_prob_plenary = plenary_model.calculate_prob_of_sentence(restored_sentence_str)
            log_prob_committee = committee_model.calculate_prob_of_sentence(restored_sentence_str)
            
            f_out.write(f"probability of plenary sentence in plenary corpus: {log_prob_plenary:.2f}\n")
            f_out.write(f"probability of plenary sentence in committee corpus: {log_prob_committee:.2f}\n")

def main(file_path, output_dir):
    output_file = os.path.join(output_dir, "knesset_collocations.txt")
    original_file = os.path.join(output_dir, "original_sampled_sents.txt")
    masked_file = os.path.join(output_dir, "masked_sampled_sents.txt")
    results_file = os.path.join(output_dir, "sampled_sents_results.txt")
    perplexity_file = os.path.join(output_dir, "perplexity_result.txt")

    # Load Corpus
    try:
        data = load_jsonl(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except ValueError:
        print(f"Error: Invalid JSON format in {file_path}")
        return
    
    # Filter Sentences by Protocol Type
    committee_sentences = get_sentences_by_type(data, 'committee')
    plenary_sentences = get_sentences_by_type(data, 'plenary')

    copy_of_committee_sentences = committee_sentences
    copy_of_plenary_sentences = plenary_sentences

    # Add start tokens to sentences
    committee_sentences = add_start_tokens(committee_sentences)
    plenary_sentences = add_start_tokens(plenary_sentences)

    # Initialize Trigram_LM Models
    committee_model = Trigram_LM(committee_sentences)
    plenary_model = Trigram_LM(plenary_sentences)

    with open(output_file, "w", encoding="utf-8") as f:
        for n in [2, 3, 4]:
            # Write the header
            if n == 2:
                f.write("Two-gram collocations:\n")
            elif n == 3:
                f.write("Three-gram collocations:\n")
            elif n == 4:
                f.write("Four-gram collocations:\n")

            # Process each metric (frequency and TF-IDF)
            for metric in ["Frequency", "TF-IDF"]:
                f.write(f"{metric}:\n")

                # Committee corpus
                f.write("Committee corpus:\n")
                committee_collocations = committee_model.get_k_n_t_collocations(
                    k=10, n=n, t=5, corpus=committee_sentences, score_type=metric
                )
                if committee_collocations:
                    for collocation in committee_collocations:
                        f.write(f"{collocation}\n")
                else:
                    f.write("No collocations found.\n")
                f.write("\n")

                # Plenary corpus
                f.write("Plenary corpus:\n")
                plenary_collocations = plenary_model.get_k_n_t_collocations(
                    k=10, n=n, t=5, corpus=plenary_sentences, score_type=metric
                )
                if plenary_collocations:
                    for collocation in plenary_collocations:
                        f.write(f"{collocation}\n")
                else:
                    f.write("No collocations found.\n")
                f.write("\n")

    # Combine all sentences for sampling
    all_sentences = copy_of_committee_sentences + copy_of_plenary_sentences

    # Tokenize sentences
    tokenized_sentences = [sentence.split() for sentence in all_sentences]

    # Sample and save sentences
    sample_and_save_sentences(
        sentences=tokenized_sentences,
        x=0.1,  # Percentage of tokens to mask
        num_samples=10,  # Number of sentences to sample
        original_file=original_file,
        masked_file=masked_file
    )

    # Task 3.3: Restore and evaluate sentences
    restore_and_evaluate_sentences(
        committee_model=committee_model,
        plenary_model=plenary_model,
        original_file=original_file,
        masked_file=masked_file,
        output_file=results_file
    )

    # Task 3.4: Calculate Perplexity for Masked Tokens
    total_perplexity = 0.0
    sentence_count = 0

    with open(original_file, 'r', encoding='utf-8') as f_orig, \
         open(masked_file, 'r', encoding='utf-8') as f_masked, \
         open(results_file, 'r', encoding='utf-8') as f_restored, \
         open(perplexity_file, 'w', encoding='utf-8') as f_out:

        original_sentences = [line.strip() for line in f_orig]
        masked_sentences = [line.strip() for line in f_masked]
        restored_sentences = [line.split("plenary_sentence: ")[1].strip() for line in f_restored if "plenary_sentence" in line]

        for orig, masked, restored in zip(original_sentences, masked_sentences, restored_sentences):
            perplexity = plenary_model.calculate_perplexity(orig, masked, restored)
            total_perplexity += perplexity
            sentence_count += 1

        # Calculate average perplexity
        avg_perplexity = total_perplexity / sentence_count if sentence_count > 0 else float('inf')
        f_out.write(f"{avg_perplexity:.2f}\n")



if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)

    # Get input folder and output file from command-line arguments
    file_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Run the main function
    main(file_path, output_dir)

