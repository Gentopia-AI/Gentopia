from typing import List
from typing import List, Union, Dict, Any, Optional
import numpy as np
import os

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    """
    Get the key for the prompt input.

    Args:
        inputs (Dict[str, Any]): The input dictionary.
        memory_variables (List[str]): List of memory variables.

    Returns:
        str: The key for the prompt input.

    Raises:
        ValueError: If more than one input key is found.
    """
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected, got {prompt_input_keys}")
    return prompt_input_keys[0]


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """
    Get a value from a dictionary or an environment variable.

    Args:
        key (str): The key to search in the dictionary.
        env_key (str): The environment variable key.
        default (Optional[str], optional): Default value if the key is not found. Defaults to None.

    Returns:
        str: The value associated with the key.

    Raises:
        ValueError: If the key is not found and no default value is provided.
    """
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable "
            f"`{env_key}` which contains it, or pass "
            f"`{key}` as a named parameter."
        )


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """
    Get a value from a dictionary or an environment variable.

    Args:
        data (Dict[str, Any]): The input dictionary.
        key (str): The key to search in the dictionary.
        env_key (str): The environment variable key.
        default (Optional[str], optional): Default value if the key is not found. Defaults to None.

    Returns:
        str: The value associated with the key.
    """
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """
    Calculate row-wise cosine similarity between two equal-width matrices.

    Args:
        X (Matrix): The first matrix.
        Y (Matrix): The second matrix.

    Returns:
        np.ndarray: The cosine similarity matrix.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])
    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. "
            f"X has shape {X.shape} and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """
    Calculate maximal marginal relevance.

    Args:
        query_embedding (np.ndarray): The query embedding.
        embedding_list (list): The list of embeddings.
        lambda_mult (float, optional): The lambda multiplier. Defaults to 0.5.
        k (int, optional): The number of embeddings to select. Defaults to 4.

    Returns:
        List[int]: The indices of the selected embeddings.

    Raises:
        ValueError: If the number of embeddings to select is invalid.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs
