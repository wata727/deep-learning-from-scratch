import numpy as np
import numpy.typing as npt

def shuffle_dataset(x: npt.NDArray, t: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    permulation = np.random.permutation(x.shape[0])
    x = x[permulation,:] if x.ndim == 2 else x[permulation,:,:,:]
    t = t[permulation]

    return x, t
