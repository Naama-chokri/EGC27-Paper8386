import numpy as np
import math
import numba


@numba.njit
def _fill_cm(m, c1, c2):
    m[:] = 0

    for a, b in zip(c1, c2):
        m[a, b] += 1


@numba.njit
def mcc(confusion_matrix):
    tp = confusion_matrix[0, 0]
    tn = confusion_matrix[1, 1]
    fp = confusion_matrix[1, 0]
    fn = confusion_matrix[0, 1]

    x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if x == 0:
        return 0
    else:
        return ((tp * tn) - (fp * fn)) / math.sqrt(x)


@numba.njit(parallel=True)
def calculate_mcc_numba_X_y_parallel(X, y):
    rows, columns = X.shape

    num_cpus = numba.get_num_threads()

    # for each thread, allocate array for confusion matrix
    cms = []
    for i in range(num_cpus):
        cms.append(np.empty((2, 2), dtype="float64"))
    # make indexes for each thread
    thread_column_idxs = np.array_split(np.arange(columns), num_cpus)

    out = np.empty(shape=columns, dtype="float64")
    for thread_idx in numba.prange(num_cpus):
        for i in thread_column_idxs[thread_idx]:
            _fill_cm(cms[thread_idx], X[:, i], y)
            out[i] = abs(mcc(cms[thread_idx]))
    return np.mean(out)


@numba.njit(parallel=True)
def calculate_mcc_numba_X_parallel(X):
    rows, columns = X.shape

    num_cpus = numba.get_num_threads()

    # for each thread, allocate array for confusion matrix
    cms = []
    for i in range(num_cpus):
        cms.append(np.empty((2, 2), dtype="float64"))

    out = np.empty(shape=(columns, columns), dtype="float64")

    # make indexes for each thread
    thread_column_idxs = np.array_split(np.arange(columns), num_cpus)

    for i in range(columns):
        c1 = X[:, i]

        for thread_idx in numba.prange(num_cpus):
            for j in thread_column_idxs[thread_idx]:
                if j < i + 1:
                    continue

                c2 = X[:, j]
                _fill_cm(cms[thread_idx], c1, c2)
                out[i, j] = abs(mcc(cms[thread_idx]))

    out2, cnt = 0.0, 0
    for i in range(columns):
        for j in range(i + 1, columns):
            out2 += out[i, j]
            cnt += 1
    if cnt == 0:
        return 0
    else:
        return out2 / cnt


def merit_calculation_for_binary_dataset_using_matthews_coef_with_numba(X, y):
    n_sample, k = X.shape
    rcf, rff = calculate_mcc_numba_X_y_parallel(X, y), calculate_mcc_numba_X_parallel(X)

    return (k * rcf) / math.sqrt(k + k * (k - 1) * rff)
