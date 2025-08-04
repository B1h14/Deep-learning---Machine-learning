import numpy as np

def softmax(x, axis=-1):
    """
    Compute the softmax of x along the specified axis.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True)) # subtract max for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
def compute_qkv(X, W_q, W_k, W_v):
    """
    Compute the query, key, and value matrices from input X using weight matrices W_q, W_k, and W_v.
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V 

def self_attention(Q, K, V):
    """
    Compute self-attention scores using query, key, and value matrices.
    """
    attention_scores = np.dot(Q, K.T) / np.sqrt(Q.shape[-1])
    attention_weights = softmax(attention_scores, axis=-1)
    output = np.dot(attention_weights, V)
    return output

def multi_head_attention(Q, K, V, n_heads):
    """
    Compute multi-head attention using query, key, and value matrices.
    """
    head_dim = Q.shape[-1] // n_heads
    outputs = []
    for i in range(n_heads):
        Q_head = Q[:, i * head_dim:(i + 1) * head_dim]
        K_head = K[:, i * head_dim:(i + 1) * head_dim]
        V_head = V[:, i * head_dim:(i + 1) * head_dim]
        output = self_attention(Q_head, K_head, V_head)
        outputs.append(output)
    return np.concatenate(outputs, axis=-1)