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
        return (
            np.array([], dtype=x.dtype),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

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
    items, run_length, _ = find_runs(x == phage_pos)
    run_length = np.array(run_length, dtype=np.str_)
    tmp = np.empty(items.shape, dtype=np.str_)
    # print(phage_pos, items, run_length)
    tmp[items != phage_pos] = "n"
    tmp[items == phage_pos] = "V"
    x = np.char.add(run_length, tmp)
    return "".join(x)


def get_window_summary(x, class_map: dict[int, str], classes: list[str]):
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

    def vmap(i: str, classes: list):
        if i.lower() in classes:
            return i[0].upper()
        return i[0].lower()

    class_sum_ = {k: vmap(v, classes=classes) for k, v in class_map.items()}
    x = x.flatten()
    items, run_length, _ = find_runs(x)
    run_length = np.array(run_length, dtype=np.str_)
    tmp = np.empty(items.shape, dtype=np.str_)
    # print(phage_pos, items, run_length)
    for k, v in class_sum_.items():
        tmp[items == k] = v
        # tmp[items == phage_pos] = "V"
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

    if x.ndim == 0:
        # Scalar binary logit.
        return -logsumexp(np.array([x, 0.0]), axis=-1)

    # Multiclass softmax case
    if x.shape[-1] == 2:
        return -logsumexp(x, axis=axis)

    # Binary case: single logit per sample.
    # log(exp(z) + 1) = logsumexp([z, 0])
    squeezed = x.squeeze(axis=-1) if x.shape[-1] == 1 else x
    return -logsumexp(np.stack([squeezed, np.zeros_like(squeezed)], axis=-1), axis=-1)


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


# --- Experimental linear-chain CRF (Viterbi) window decoding -----------------
#
# ``smoothen_scores`` above is unused; CRF decoding below supersedes it.
#
# A contig's window labels are decoded jointly:
#     y_hat = argmax_y  sum_t s_t(y_t) - lambda * sum_t P(y_{t-1}, y_t)
# where s_t is the per-window log-softmax of the network logits and P is a
# transition-prior matrix (0 on the diagonal). No training is involved; the
# prior is fixed a priori from biological class co-occurrence plausibility.

#: Biological co-occurrence prior tiers for the transition-cost matrix, keyed
#: by lower-cased class name. Pairs not listed stay at the neutral cost 1.0.
#: 0.5 = plausible on one contig (prophages, plasmids, eukaryotic viruses),
#: 3.0 = implausible (cellular domain switches, phages in eukaryotes).
_CRF_PRIOR_TIERS: tuple = (
    (
        0.5,
        (
            ("bacteria", "phage"),
            ("bacteria", "plasmid"),
            ("archaea", "phage"),
            ("archaea", "plasmid"),
            ("phage", "plasmid"),
            ("eukarya", "virus"),
        ),
    ),
    (
        3.0,
        (
            ("bacteria", "eukarya"),
            ("archaea", "eukarya"),
            ("bacteria", "archaea"),
            ("eukarya", "phage"),
            ("eukarya", "plasmid"),
        ),
    ),
)


def default_transition_prior(class_names: list[str]) -> np.ndarray:
    """
    Builds the default biological transition-prior matrix P for CRF decoding.

    Costs are symmetric with a zero diagonal; pairs not listed in
    ``_CRF_PRIOR_TIERS`` get the neutral cost 1.0. Class names not present in
    ``class_names`` are skipped, so models with fewer classes (e.g. 4-class or
    binary heads) degrade gracefully to a uniform Potts prior.

    Args:
    ----
        class_names (list[str]): Class names in index order.

    Returns:
    -------
        np.ndarray: Prior matrix P of shape (len(class_names), len(class_names)).
    """

    names = [str(n).lower() for n in class_names]
    n = len(names)
    prior = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(prior, 0.0)
    for value, pairs in _CRF_PRIOR_TIERS:
        for a, b in pairs:
            if a in names and b in names:
                i, j = names.index(a), names.index(b)
                prior[i, j] = value
                prior[j, i] = value
    return prior


def build_transition_costs(
    class_names: list[str],
    switch_cost: float,
    prior: str = "biological",
    user_matrix: dict | None = None,
) -> np.ndarray:
    """
    Assembles the CxC transition-cost matrix ``lambda * P`` for CRF decoding.

    Args:
    ----
        class_names (list[str]): Class names in index order.
        switch_cost (float): Global transition cost lambda (log-probability
                             units).
        prior (str): "biological" (default tier table) or "uniform" (same cost
                     for every class switch, i.e. plain Potts smoothing).
        user_matrix (dict | None): Optional custom costs keyed by class name,
                     e.g. ``{"bacteria": {"phage": 0.5}}``. Entries are applied
                     symmetrically; unspecified pairs stay neutral. Overrides
                     ``prior``.

    Returns:
    -------
        np.ndarray: Cost matrix of shape (len(class_names), len(class_names)).
    """

    names = [str(n).lower() for n in class_names]
    n = len(names)
    if user_matrix:
        p = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(p, 0.0)
        for a, row in user_matrix.items():
            a = str(a).lower()
            if a not in names or not isinstance(row, dict):
                continue
            for b, value in row.items():
                b = str(b).lower()
                if b not in names:
                    continue
                i, j = names.index(a), names.index(b)
                p[i, j] = float(value)
                p[j, i] = float(value)
        np.fill_diagonal(p, 0.0)
    elif prior == "uniform":
        p = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(p, 0.0)
    else:
        p = default_transition_prior(names)
    return float(switch_cost) * p


def viterbi_decode(
    logits: np.ndarray,
    switch_cost: float = 2.0,
    transition_costs: np.ndarray | None = None,
) -> np.ndarray:
    """
    MAP-decodes a contig's window class sequence with a linear-chain CRF.

    Emissions are the per-window log-softmax of the network logits; switching
    from class a to class b between adjacent windows costs
    ``transition_costs[a, b]`` (uniform ``switch_cost`` off-diagonal when not
    given). Solved exactly with the Viterbi algorithm in O(T * C^2).

    Args:
    ----
        logits (np.ndarray): Per-window logits, shape (T, C).
        switch_cost (float): Uniform transition cost lambda, used only when
                             ``transition_costs`` is None. 0.0 reproduces
                             independent per-window argmax.
        transition_costs (np.ndarray | None): Full CxC cost matrix (lambda * P)
                             as built by :func:`build_transition_costs`.

    Returns:
    -------
        np.ndarray: Decoded class indices, int array of shape (T,).
    """

    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    t_len, n_classes = z.shape
    emissions = z - logsumexp(z, axis=-1)[:, None]
    if t_len == 1 or n_classes == 1:
        return np.argmax(emissions, axis=-1)
    if transition_costs is None:
        costs = np.full((n_classes, n_classes), float(switch_cost))
        np.fill_diagonal(costs, 0.0)
    else:
        costs = np.asarray(transition_costs, dtype=np.float64)
    delta = np.empty((t_len, n_classes))
    backptr = np.empty((t_len, n_classes), dtype=np.int64)
    delta[0] = emissions[0]
    for t in range(1, t_len):
        # scores[prev, cur]: best path score arriving at cur from prev
        scores = delta[t - 1][:, None] - costs
        backptr[t] = np.argmax(scores, axis=0)
        delta[t] = emissions[t] + scores[backptr[t], np.arange(n_classes)]
    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(t_len - 2, -1, -1):
        path[t] = backptr[t + 1][path[t + 1]]
    return path


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
