import numpy as np
import numpy.typing as npt

def shuffle_dataset[T: np.generic, U: np.generic](x: npt.NDArray[T], t: npt.NDArray[U]) -> tuple[npt.NDArray[T], npt.NDArray[U]]:
    permulation = np.random.permutation(x.shape[0])
    x = x[permulation,:] if x.ndim == 2 else x[permulation,:,:,:]
    t = t[permulation]

    return x, t
