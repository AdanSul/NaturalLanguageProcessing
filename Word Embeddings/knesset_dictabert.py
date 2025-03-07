from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as transformers_logging
import os
import sys
import warnings

# Suppress warnings and logging
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
transformers_logging.set_verbosity_error()

def load_dictabert_model():
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("dicta-il/dictabert", trust_remote_code=True)
    return tokenizer, model

def process_masked_sentences(masked_file, tokenizer, model):
    results = []

    with open(masked_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Replace [*] with [MASK]
            masked_sentence = line.replace("[*]", "[MASK]")

            # Tokenize the sentence
            inputs = tokenizer(masked_sentence, return_tensors="pt")
            mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

            # Predict the masked tokens
            outputs = model(**inputs)
            predictions = outputs.logits

            # Get the predicted token for each mask
            predicted_tokens = []
            for mask_index in mask_token_index:
                predicted_token_id = predictions[0, mask_index].argmax(dim=-1).item()
                predicted_tokens.append(tokenizer.decode(predicted_token_id).strip())

            # Reconstruct the sentence
            reconstructed_sentence = masked_sentence
            for token in predicted_tokens:
                reconstructed_sentence = reconstructed_sentence.replace("[MASK]", token, 1)

            # Append result
            results.append({
                "masked_sentence": line,
                "dictaBERT_sentence": reconstructed_sentence,
                "dictaBERT_tokens": predicted_tokens
            })

    return results


def save_results(results, output_dir):
    output_file = os.path.join(output_dir, "dictabert_results.txt")

    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(f"masked_sentence: {result['masked_sentence']}\n")
            file.write(f"dictaBERT_sentence: {result['dictaBERT_sentence']}\n")
            file.write(f"dictaBERT tokens: {', '.join(result['dictaBERT_tokens'])}\n")


def main(masked_file, output_dir):
    # Load DictaBERT model and tokenizer
    tokenizer, model = load_dictabert_model()
    # Process masked sentences
    results = process_masked_sentences(masked_file, tokenizer, model)
    # Save the results
    save_results(results, output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    # Get input folder and output file from command-line arguments
    masked_file = sys.argv[1]
    output_dir = sys.argv[2]

    # masked_file = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\HW2\masked_sampled_sents.txt'
    # output_dir = r'C:\Users\adans\OneDrive\שולחן העבודה\Courses\Natural_Language_Processing\HW4'

    main(masked_file, output_dir)