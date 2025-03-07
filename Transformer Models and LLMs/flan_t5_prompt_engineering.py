import sys
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
from datasets import load_dataset

def zero_shot_prompt(review):
    return f"Classify the sentiment of the following movie review as either 'positive' or 'negative' '{review}'"

def few_shot_prompt(review):
    return (
        "Classify the sentiment of the following movie review as either 'positive' or 'negative'.\n\n"
        "Examples:\n"
        "1. 'I absolutely loved this movie. The story was gripping, and the acting was phenomenal.' Sentiment: positive.\n"
        "2. 'This was the worst film I've ever seen. The plot was boring, and the characters were flat.' Sentiment: negative.\n"
        f"Now, classify this review: '{review}'\n"
        "Sentiment:"
    )

def instruction_based_prompt(review):
    return (
        "You are a sentiment analysis model. Your task is to classify movie reviews as 'positive' or 'negative' based on their content.\n"
        "Ensure your classification is accurate and reflects the overall sentiment of the review.\n\n"
        f"Review: {review}\n"
        "Sentiment:"
    )

def main():
    if len(sys.argv) != 3:
        raise ValueError("Usage: python flan_t5_prompt_engineering.py <path/to/imdb_subset> <path/to/flan_t5_imdb_results.txt>")
        sys.exit(1)
    
    imdb_subset_path = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load the dataset
    # dataset = load_from_disk(imdb_subset_path)
    try:
        dataset = load_from_disk(imdb_subset_path)
    except Exception:
        dataset = load_dataset("imdb")["train"]
    
    # Stratified sampling: 25 positive and 25 negative reviews
    positive_reviews = dataset.filter(lambda x: x['label'] == 1).shuffle(seed=42).select(range(25))
    negative_reviews = dataset.filter(lambda x: x['label'] == 0).shuffle(seed=42).select(range(25))
    
    # Combine and shuffle with seed
    subset = concatenate_datasets([positive_reviews, negative_reviews]).shuffle(seed=42)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    results = []
    
    for example in subset:
        review = example['text']
        true_label = "positive" if example['label'] == 1 else "negative"
        
        # Zero-shot
        zero_shot_input = tokenizer.encode(
            zero_shot_prompt(review),
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        zero_shot_output = model.generate(zero_shot_input)
        zero_shot_result = tokenizer.decode(zero_shot_output[0], skip_special_tokens=True).strip().lower()
        
        # Few-shot
        few_shot_input = tokenizer.encode(
            few_shot_prompt(review),
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        few_shot_output = model.generate(few_shot_input)
        few_shot_result = tokenizer.decode(few_shot_output[0], skip_special_tokens=True).strip().lower()
        
        # Instruction-based
        instruction_input = tokenizer.encode(
            instruction_based_prompt(review),
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        instruction_output = model.generate(instruction_input)
        instruction_result = tokenizer.decode(instruction_output[0], skip_special_tokens=True).strip().lower()
        
        # Save results
        results.append({
            "review": review,
            "true_label": true_label,
            "zero_shot": zero_shot_result,
            "few_shot": few_shot_result,
            "instruction_based": instruction_result,
        })
    
    # # Save results to a file
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for i, result in enumerate(results, 1):
    #         f.write(f"Review {i}: {result['review']}\n")
    #         f.write(f"Review {i} true label: {result['true_label']}\n")
    #         f.write(f"Review {i} zero-shot: {result['zero_shot']}\n")
    #         f.write(f"Review {i} few-shot: {result['few_shot']}\n")
    #         f.write(f"Review {i} instruction-based: {result['instruction_based']}\n")
    #         f.write("\n")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for i, result in enumerate(results, 1):
                f.write(f"Review {i}: {result['review']}\n")
                f.write(f"Review {i} true label: {result['true_label']}\n")
                f.write(f"Review {i} zero-shot: {result['zero_shot']}\n")
                f.write(f"Review {i} few-shot: {result['few_shot']}\n")
                f.write(f"Review {i} instruction-based: {result['instruction_based']}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")
    
    # Calculate accuracy
    true_labels = [res['true_label'] for res in results]
    zero_shot_predictions = [res['zero_shot'] for res in results]
    few_shot_predictions = [res['few_shot'] for res in results]
    instruction_predictions = [res['instruction_based'] for res in results]
    
    zero_shot_acc = accuracy_score(true_labels, zero_shot_predictions)
    few_shot_acc = accuracy_score(true_labels, few_shot_predictions)
    instruction_acc = accuracy_score(true_labels, instruction_predictions)
    
    # print(f"Zero-shot accuracy: {zero_shot_acc:.2f}")
    # print(f"Few-shot accuracy: {few_shot_acc:.2f}") 
    # print(f"Instruction-based accuracy: {instruction_acc:.2f}")

if __name__ == "__main__":
    main()
