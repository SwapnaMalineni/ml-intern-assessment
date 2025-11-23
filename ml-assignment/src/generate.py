import os
from ngram_model import TrigramModel

def main():
    """
    Main function to train the trigram model and generate text.
    """
    # Construct the path to the corpus file relative to this script
    # This makes the script runnable from any directory
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(base_dir, '..', 'data', 'example_corpus.txt')
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    # Initialize and train the model
    model = TrigramModel()
    model.fit(text)

    # Generate new text
    print("Generating text from the model...")
    generated_text = model.generate(max_length=30)

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------\n")

if __name__ == "__main__":
    main()