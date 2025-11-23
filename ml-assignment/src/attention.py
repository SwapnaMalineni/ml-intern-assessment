import numpy as np

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the scaled dot-product attention.

    Args:
        q (np.ndarray): Query matrix, shape (..., seq_len_q, d_k)
        k (np.ndarray): Key matrix, shape (..., seq_len_k, d_k)
        v (np.ndarray): Value matrix, shape (..., seq_len_v, d_v)
                        Note: seq_len_k and seq_len_v must be the same.
        mask (np.ndarray, optional): Mask to apply to the attention scores.
                                     Shape (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        tuple: A tuple containing:
            - output (np.ndarray): The attention output, shape (..., seq_len_q, d_v)
            - attention_weights (np.ndarray): The attention weights, shape (..., seq_len_q, seq_len_k)
    """
    # Matmul Q and K transpose
    # The ... allows this to work with batch and head dimensions
    matmul_qk = np.matmul(q, k.swapaxes(-2, -1))

    # Scale matmul_qk
    d_k = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # Apply mask if provided (add a large negative number)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to get attention weights
    attention_weights = softmax(scaled_attention_logits, axis=-1)

    # Matmul attention weights and V
    output = np.matmul(attention_weights, v)

    return output, attention_weights