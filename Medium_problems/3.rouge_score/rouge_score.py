def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    # Your code here
    tokens_ref = reference.split()
    tokens_cand = candidate.split()
    tokens_ref_dict = {}
    tokens_cand_dict = {}
    for token in tokens_ref:
        tokens_ref_dict[token] = tokens_ref_dict.get(token, 0) + 1
    overlap = 0
    for token in tokens_cand:
        tokens_cand_dict[token] = tokens_cand_dict.get(token, 0) + 1
    for token in tokens_ref_dict.keys():
        overlap += min(tokens_ref_dict[token], tokens_cand_dict.get(token, 0))
    precision = overlap / len(tokens_cand) if len(tokens_cand) > 0 else 0
    recall = overlap / len(tokens_ref) if len(tokens_ref) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }