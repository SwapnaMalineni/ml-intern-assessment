# Evaluation

## Task 1: Trigram Language Model

### Storage of N-Gram Counts
The trigram counts are stored using a nested dictionary structure, specifically a `defaultdict` of `Counter` objects:  
```python
self.trigram_counts = defaultdict(Counter)
```
This allows efficient storage and retrieval of the counts of the third word given a bigram context `(w1, w2)`.

### Text Cleaning, Padding, and Unknown Words
- Input text is converted to lowercase for case insensitivity.  
- Punctuation is removed while preserving apostrophes inside words to maintain contractions.  
- The text is tokenized by whitespace.  
- Padding is done with two start tokens `<s> <s>` at the beginning and one end token `<e>` at the end to capture sentence boundaries properly in trigram context.

Unknown words are implicitly handled by the model during generation, as any unseen trigram context will lead to no continuation, causing generation to stop.

### Generate Function and Probabilistic Sampling
- Generation starts with the initial bigram of start tokens.  
- For each step, the next word is sampled probabilistically according to the distribution of counts for that bigram context using `random.choices` weighted by counts.  
- Generation stops when the end token `<e>` is sampled or the max length is reached.  
- This ensures variety in generated text rather than always picking the most likely next word.

### Other Design Decisions
- The use of `defaultdict(Counter)` simplifies count bookkeeping.  
- Regular expressions were designed to carefully clean the text while preserving meaningful apostrophes.  
- Careful padding of start and end tokens enables context-aware trigram modeling at sentence boundaries.

## Task 2: Scaled Dot-Product Attention

### Implementation
- Implemented purely with numpy for all numerical operations, abiding by the assignment restrictions.  
- The attention score matrix is computed as the dot product of queries (Q) and keys (K), transposed:  
  \[
  \text{scores} = Q K^T
  \]
- Scores are scaled by \(\frac{1}{\sqrt{d_k}}\) to stabilize gradients and improve learning dynamics.  
- An optional mask can be added to prevent attention to certain positions, applied by adding a large negative value to those logits before softmax.  
- Softmax is applied row-wise on scaled scores to compute attention weights (probabilities).  
- The output is the weighted sum of the value (V) vectors using the attention weights.

### Demonstration
- A separate script creates example Q, K, V matrices with fixed dimensions and random values for reproducibility.  
- It calculates the attention outputs and prints input shapes, a sample of output values, and the attention weights for inspection.  
- This script verifies correctness and provides a clear example for understanding the mechanism.

---

This design satisfies the fundamental requirements for both tasks, providing a clean, documented, and functional implementation with testing and demonstration where applicable.
