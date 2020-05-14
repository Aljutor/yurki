import array
import numpy as np
import scipy.sparse
from typing import List, Tuple, Dict


from . import yurki  # type: ignore


def tokenize_string(
    data: List[str],
    ngram: Tuple[int, int] = (2, 2),
    jobs: int = 1,
    inplace: bool = False,
) -> Tuple[scipy.sparse.csr_matrix, Dict[str, int]]:

    result, vocab = yurki.tokenize_string(data, ngram, jobs, inplace)

    # refactor in future for more optimal sparse matrix creation
    indices = array.array("i")
    values = array.array("i")
    indptr = array.array("i")
    indptr.append(0)

    for i, row in enumerate(result):
        indices.extend(row[0])
        values.extend(row[1])
        indptr.append(len(indices))

    indices = np.frombuffer(indices, dtype=np.int32)
    indptr = np.frombuffer(indptr, dtype=np.int32)
    values = np.frombuffer(values, dtype=np.intc)

    m = scipy.sparse.csr_matrix(
        (values, indices, indptr), shape=(len(result), len(vocab)), dtype=np.int32
    )
    return m, vocab
