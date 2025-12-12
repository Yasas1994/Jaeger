import numpy as np
import logging
from jaeger.utils.misc import safe_divide

logger = logging.getLogger("jaeger")


def find_runs(x):  # sourcery skip: extract-method
    # from https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """
    Find runs of consecutive identical items in a 1D array.

    Args:
        x (list or np.array): A list or numpy array of integers.

    Returns:
        tuple: (run_values, run_lengths, run_starts)
    """
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("Only 1D arrays are supported")

    n = x.shape[0]
    if n == 0:
        return np.array([], dtype=x.dtype), np.array([], dtype=int), np.array([], dtype=int)

    # Find where the value changes
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    run_values = x[run_starts]
    run_lengths = np.diff(np.append(run_starts, n))

    return run_values, run_lengths, run_starts


def get_window_summary_legacy(x, phage_pos):
    """
    returns string representation of window-wise predictions

    Args
    ----

    x : list or np.array
        list or numpy array with window-wise integer class labels

    phage_pos : int
        integer value representing Phage or Virus class

    Returns
    -------
    a string with Vs and ns. Vs represent virus or phage windows. ns
    represent cellular windows

    """
    x = x.flatten() 
    items, run_length, _ = find_runs(x== phage_pos)
    run_length = np.array(run_length, dtype=np.str_)
    tmp = np.empty(items.shape, dtype=np.str_)
    # print(phage_pos, items, run_length)
    tmp[items != phage_pos] = "n"
    tmp[items == phage_pos] = "V"
    x = np.char.add(run_length, tmp)
    return "".join(x)

def get_window_summary(x, class_map:dict[int,str], classes:list[str]):
    """
    returns string representation of window-wise predictions

    Args
    ----

    x : list or np.array
        list or numpy array with window-wise integer class labels

    phage_pos : int
        integer value representing Phage or Virus class

    Returns
    -------
    a string with Vs and ns. Vs represent virus or phage windows. ns
    represent cellular windows

    """
    def vmap(i: str, classes:list):
        if i.lower() in classes:
            return i[0].upper()
        return i[0].lower()
    class_sum_ = {k: vmap(v, classes=classes) for k, v in class_map.items()}
    x = x.flatten() 
    items, run_length, _ = find_runs(x)
    run_length = np.array(run_length, dtype=np.str_)
    tmp = np.empty(items.shape, dtype=np.str_)
    # print(phage_pos, items, run_length)
    for k,v in class_sum_.items():
        tmp[items == k] = v
        #tmp[items == phage_pos] = "V"
    x = np.char.add(run_length, tmp)
    return "".join(x)

def update_dict(x, num_classes=4):
    # sourcery skip: remove-redundant-constructor-in-dict-union
    """
    Updates a dictionary with key-value pairs from input data.

    Args:
    ----
        x: Tuple containing keys and values to update the dictionary.
        num_classes (int, optional): Number of classes for initializing
                                     the dictionary keys. Defaults to 4.

    Returns:
    -------
        None: The dictionary is updated in place.
    """

    return {i: 0 for i in range(num_classes)} | dict(zip(x[0], x[1]))


def shanon_entropy(p):
    """
    Calculates the Shannon entropy (information gain) of a probability
    distribution.

    Args:
    ----
        p (array-like): The probability distribution as an array-like object.

    Returns:
    -------
        float: The Shannon entropy value calculated from the input probability
        distribution.
    """

    p = np.array(p)
    result = np.where(p > 0.0000000001, p, -10)
    p_log = np.log2(result, out=result, where=result > 0)
    return -np.sum(p * p_log, axis=-1)


# def softmax_entropy(x):
#     """
#     Calculates the entropy of a softmax output distribution.

#     Args:
#     ----
#         x (array-like): The softmax output distribution as an array-like
#                         object.

#     Returns:
#     -------
#         float: The entropy value calculated from the softmax output
#                distribution.
#     """

#     ex = np.exp(x)
#     return shanon_entropy(ex / np.sum(ex, axis=-1).reshape(-1, 1))

def binary_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def softmax_entropy(p, axis=-1, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=axis)


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable logsumexp implementation in NumPy.
    """
    xmax = np.max(x, axis=axis, keepdims=True)
    stable = x - xmax
    return xmax.squeeze(axis=axis) + np.log(np.sum(np.exp(stable), axis=axis))

def energy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Energy score from logits.

    Parameters
    ----------
    x : np.ndarray
        - Binary: logits of shape (...)
        - Multiclass: logits of shape (..., C)
    axis : int
        Class axis for multiclass logits.

    Returns
    -------
    np.ndarray
        Energy values (lower = more confident).
    """
    x = np.asarray(x, dtype=np.float64)

    # Binary case: single logit per sample
    if x.ndim == 0 or (x.ndim >= 1 and x.shape[-1] != 2):
        # log(exp(z) + 1) = logsumexp([z, 0])
        return -logsumexp(np.stack([x, np.zeros_like(x)], axis=-1), axis=-1)

    # Multiclass softmax case
    return -logsumexp(x, axis=axis)


def sigmoid(x):
    """
    Calculates sigmoid.

    Args:
    ----
        x (array-like): logits as an array-like
                        object.

    Returns:
    -------
        float: sigmoid transformed logits
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Calculates softmax of output distribution.

    Args:
    ----
        x (array-like): logits as an array-like
                        object.

    Returns:
    -------
        float: softmax output distribution.
    """

    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1).reshape(-1, 1)


def smoothen_scores(x, w=5):
    """
    Smoothes the scores of different classes using a moving average.

    Args:
    ----
        x (array-like): The input array containing scores for different
                        classes.
        w (int, optional): The window size for the moving average.
                           Defaults to 5.

    Returns:
    -------
        array-like: The smoothed scores for each class after applying the
                    moving average.
    """
    return np.column_stack(
        [np.convolve(x[:, i], np.ones(w) / w, mode="same") for i in range(x.shape[1])]
    )


def ood_predict(x_features, params):
    """
    Predicts out-of-distribution (OOD) probabilities using logistic regression
    parameters.

    Args:
    ----
        x_features (array-like): The input features to predict OOD
                                 probabilities.
        params (dict): Dictionary containing logistic regression parameters.

    Returns:
    -------
        tuple: A tuple containing the predicted OOD probabilities and logits.
    """

    # Normalize x_features using NumPy's built-in functions
    x_features = (x_features - np.mean(x_features, axis=-1, keepdims=True)) / np.std(
        x_features, axis=-1, keepdims=True
    )
    logits = np.dot(x_features, params["coeff"].flatten()) + params["intercept"]
    return (1 / (1 + np.exp(logits))).flatten(), logits


def normalize(x):
    """
    Normalizes the input array along axis 1 using mean and standard deviation.

    Args:
    ----
        x (array-like): The input array to be normalized.

    Returns:
    -------
        array-like: The normalized array after subtracting the mean and
                    dividing by the standard deviation along axis 1.
    """

    x_mean = x.mean(axis=1).reshape(-1, 1)
    x_std = x.std(axis=1).reshape(-1, 1)
    return (x - x_mean) / x_std


def normalize_with_batch_stats(x, mean, std):
    """
    Normalizes the input array using batch mean and standard deviation.

    Args:
    ----
        x (array-like): The input array to be normalized.
        mean (array-like): The batch mean values.
        std (array-like): The batch standard deviation values.

    Returns:
    -------
        array-like: The normalized array using the provided batch mean and
                    standard deviation.
    """

    return (x - mean) / std


def normalize_l2(x):
    """
    Normalizes the input array along axis 1 using L2 norm.

    Args:
    ----
        x (array-like): The input array to be normalized.

    Returns:
    -------
        array-like: The L2 normalized array along axis 1.
    """

    return x / np.linalg.norm(x, 2, axis=1).reshape(-1, 1)


def ood_predict_default(x_features, params):
    """
    Predicts out-of-distribution (OOD) probabilities using logistic regression
    or a saved sklearn model.

    Args:
    ----
        x_features (array-like): The input features to predict OOD
                                 probabilities.
        params (dict): Dictionary containing parameters for prediction.

    Returns:
    -------
        tuple: A tuple containing the predicted OOD probabilities and logits
        based on the specified method.
    """

    # use parameters extimated using sklearn
    if params["type"] == "params":
        # x_features = normalize_with_batch_stats(x_features,
        # params['batch_mean']),
        # params['batch_std']
        x_features = normalize(x_features)
        logits = (
            np.dot(x_features, params["coeff"].reshape(-1, 1)) + params["intercept"]
        )
        return (1 / (1 + np.exp(-logits))).flatten(), logits
    # use a saved a sklearn model
    elif params["type"] == "sklearn":
        features_data = normalize_with_batch_stats(
            x_features, params["batch_mean"], params["batch_std"]
        )
        features_data_l2 = normalize_l2(features_data)

        return params["model"].predict_proba(features_data_l2)[:, 0], 0


def get_ood_probability(ood, threshold=0.5):
    """
    Calculates the out-of-distribution (OOD) summary for alll windows by
    calculating the percentage of windows below a user-defined threshold.

    Args:
    ----
        ood (array-like): The input array of OOD values.
        threshold (float) : OOD probability threshold

    Returns:
    -------
        str: The OOD probability rounded to 2 decimal places if ood is not
             None, otherwise returns "-".
    """

    return f"{sum((ood < threshold) * 1) / len(ood):2f}" if ood is not None else "-"


def consecutive(data, stepsize=1):
    """
    Splits an array into subarrays where elements are consecutive.

    Args:
    ----
        data (array-like): The input array to split into consecutive subarrays.
        stepsize (int, optional): The step size between consecutive elements.
                                  Defaults to 1.

    Returns:
    -------
        list: A list of subarrays where elements are consecutive.
    """

    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def merge_overlapping_ranges(intervals):
    """
    Merge overlapping ranges in a list of intervals.

    Args
    ----
        intervals: List of intervals, each represented as [start, end].

    Returns
    -------
        merged_intervals: List of merged intervals.
    """
    if len(intervals) == 0:
        return []

    # Sort the intervals based on the start value
    sorted(intervals, key=lambda x: x[0])

    merged_intervals = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged_intervals[-1]

        if current_start <= last_end:  # Overlapping intervals
            merged_intervals[-1][1] = max(last_end, current_end)
        else:  # Non-overlapping intervals
            merged_intervals.append([current_start, current_end])

    return merged_intervals


def check_middle_number(array):
    """
    Checks the middle number in a given array based on specific conditions.

    Args:
    ----
        array (array-like): The input array to check for the middle number.

    Returns:
    -------
        array-like: A boolean mask indicating the middle number based on
                    certain conditions.
    """
    indices = np.arange(len(array) - 1)
    tmp = indices + np.argmax(array[indices : indices + 2], axis=1)
    mask = np.zeros(len(array), dtype=np.bool_)
    mask[tmp] = 1

    return mask


def scale_range(input, min: float, max: float):
    """
    Scales the input array to a specified range using min-max scaling.

    Args:
    ----
        input (array-like): The input array to be scaled.
        min (float): The minimum value of the output range.
        max (float): The maximum value of the output range.

    Returns:
    -------
        array-like: The scaled array within the specified range.
    """

    # min-max scaling
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def gc_skew(seq: str, window: int = 2048):
    """
    Calculates the GC skew along a DNA sequence with a specified window size.

    Args:
    ----
        seq (str): The DNA sequence for GC skew calculation.
        window (int, optional): The window size for calculating GC skew.
                                Defaults to 2048.

    Returns:
    -------
        dict: A dictionary containing the GC skew, position, and cumulative
              GC skew.
    """

    gc_skew = []
    lengths = []

    # lagging strand with negative GC skew.
    for i in range(0, len(seq) - window + 1, window):
        g = seq.count("G", i, i + window)
        c = seq.count("C", i, i + window)
        gc_skew.append(safe_divide((g - c), (g + c)))
        lengths.append(i)
    gc_skew = scale_range(
        np.convolve(np.array(gc_skew), np.ones(10) / 10, mode="same"), min=-1, max=1
    )
    cumsum = scale_range(np.cumsum(gc_skew), min=-1, max=1)
    return {"gc_skew": gc_skew, "position": np.array(lengths), "cum_gc": cumsum}


def calculate_gc_content(sequence):
    """
    Calculates the GC content of a given DNA sequence.

    Args:
    ----
        sequence (str): The DNA sequence for which GC content is calculated.

    Returns:
    -------
        float: The GC content of the DNA sequence.
    """

    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def calculate_percentage_of_n(sequence):
    """
    Calculates the percentage of 'N' bases in a given DNA sequence.

    Args:
    ----
        sequence (str): The DNA sequence

    Returns:
    -------
        float: proportion of Ns
    """

    return sequence.count("N") / len(sequence)
