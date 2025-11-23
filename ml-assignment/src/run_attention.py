import numpy as np
from attention import scaled_dot_product_attention

def main():
    """
    Demonstrates the scaled_dot_product_attention function.
    """
    # Let's define dimensions for a simple example
    seq_len = 4  # Number of tokens in the sequence
    d_k = 8      # Dimension of Query and Key vectors
    d_v = 10     # Dimension of Value vectors

    print(f"Sequence Length: {seq_len}, Key/Query Dim: {d_k}, Value Dim: {d_v}\n")

    # Create random Q, K, V matrices
    # In a real model, these would be learned projections of the input embeddings
    np.random.seed(42) # for reproducibility
    q = np.random.rand(seq_len, d_k)
    k = np.random.rand(seq_len, d_k)
    v = np.random.rand(seq_len, d_v)

    print("--- Input Matrices ---")
    print(f"Query (Q) shape: {q.shape}")
    print(f"Key (K) shape:   {k.shape}")
    print(f"Value (V) shape: {v.shape}\n")

    # Calculate attention
    output, attention_weights = scaled_dot_product_attention(q, k, v)

    print("--- Output ---")
    print(f"Attention Output shape: {output.shape}")
    print("Output (first 5 values of the first token's vector):")
    print(output[0, :5])
    print("\nAttention Weights shape:", attention_weights.shape)
    print("Attention weights (how much each token attends to others):")
    print(np.round(attention_weights, 2))

if __name__ == "__main__":
    main()