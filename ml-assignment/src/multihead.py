import numpy as np
from attention import scaled_dot_product_attention

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            d_model (int): The total dimension of the model's embeddings.
            num_heads (int): The number of attention heads.
        """
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        # Dimension of each head's key, query, and value vectors
        self.depth = d_model // num_heads

        # In a real framework, these would be learnable nn.Linear layers
        # We simulate them with random weight matrices.
        self.wq = np.random.rand(d_model, d_model)
        self.wk = np.random.rand(d_model, d_model)
        self.wv = np.random.rand(d_model, d_model)

        # Final linear layer
        self.dense = np.random.rand(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (num_heads, depth).
        Transposes the result to be (batch_size, num_heads, seq_len, depth)
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def call(self, v, k, q, mask=None):
        """
        Performs the forward pass for Multi-Head Attention.
        """
        batch_size = q.shape[0]

        # 1. Pass inputs through linear layers (projections)
        q = q @ self.wq  # (batch_size, seq_len, d_model)
        k = k @ self.wk  # (batch_size, seq_len, d_model)
        v = v @ self.wv  # (batch_size, seq_len, d_model)

        # 2. Split the d_model dimension into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 3. Apply Scaled Dot-Product Attention to each head
        # The function works on the last two dimensions (seq_len, depth)
        # and broadcasts over the batch and head dimensions.
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)

        # 4. Concatenate heads and pass through a final linear layer
        # Transpose back to (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
        # Reshape to (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = concat_attention @ self.dense  # (batch_size, seq_len_q, d_model)
        return output, attention_weights