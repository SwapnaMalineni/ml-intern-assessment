import random
import re
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Data structure to hold counts of trigrams:
        # { (w1, w2): {w3: count, ...}, ... }
        self.trigram_counts = defaultdict(Counter)
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "<e>"

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # Reset counts
        self.trigram_counts = defaultdict(Counter)

        # Clean text: lowercase and remove punctuation except apostrophes inside words
        text = text.lower()
        # Preserve apostrophes inside words, remove others. Then remove non-alphanumeric chars.
        text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text) # Remove apostrophes not inside words
        text = re.sub(r"[^\w\s']", " ", text) # Remove remaining punctuation
        # Tokenize by splitting on whitespace
        tokens = text.split()

        # Pad with start tokens and an end token
        tokens = [self.START_TOKEN, self.START_TOKEN] + tokens + [self.END_TOKEN]

        # Count trigrams
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigram_counts[(w1, w2)][w3] += 1

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        # Handle case where no model is trained or empty trigram_counts
        if not self.trigram_counts:
            return ""

        # Start with two start tokens
        current_bigram = (self.START_TOKEN, self.START_TOKEN)
        result_words = []

        for _ in range(max_length):
            next_words = self.trigram_counts.get(current_bigram)
            if not next_words:
                break

            # Probabilistically choose the next word based on counts
            population, weights = zip(*next_words.items())
            next_word = random.choices(population, weights=weights, k=1)[0]

            if next_word == self.END_TOKEN:
                break

            result_words.append(next_word)
            current_bigram = (current_bigram[1], next_word)

        return " ".join(result_words)
