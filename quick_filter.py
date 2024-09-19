from quick_tree import QuickTree
import numpy as np

class QuickFilterError(Exception):
    """
    Custom exception for QuickFilter-related errors.
    """
    pass

def add_edges(arr, window_size, mode='constant', cval=0.0):
    half_window_size = window_size // 2
    new_arr = np.zeros(window_size + len(arr))
    if window_size % 2 == 0:
        new_arr[half_window_size:-half_window_size] = arr
    else:
        new_arr[half_window_size:-half_window_size] = arr

    match mode:
        case 'nearest':
            # Repeat the nearest value at the edges
            pair = arr[0], arr[-1]

        case 'reflect':
            # Reflect the edge region symmetrically
            pair = arr[:half_window_size][::-1], arr[-half_window_size:][::-1]

        case 'mirror':
            # TODO: Make this the same as Scipy's mirror mode
            pair = arr[:half_window_size][::-1], arr[-half_window_size:][::-1]

        case 'constant':
            # Use a constant value at the edges
            pair = cval, cval

        case 'wrap':
            # Wrap around the signal, using the opposite edge values
            pair = arr[-half_window_size:], arr[:half_window_size]

        case _:
            raise QuickFilterError(f"Got invalid edge-handling mode: {mode}")

    new_arr[:half_window_size] = pair[0]
    new_arr[-half_window_size:] = pair[1]

    return new_arr

def make_output_array(output, out_len):
    """
    Returns an output array for storing results, either using a provided array 
    or creating a new one if no valid array is supplied.

    Parameters:
    ----------
    output : numpy.ndarray or None
        The user-supplied array to store the output, or None to request
        automatic creation of a new array.
    out_len : int
        The expected length of the output array.

    Returns:
    -------
    numpy.ndarray
        The output array.

    Raises:
    -------
    QuickFilterError
        If a user-supplied output array is given but its length does not match the 
        expected length.

    Example:
    --------
    user_array = np.zeros(5)
    out = make_output_array(user_array, 5)  # Uses the provided array
    out = make_output_array(None, 5)        # Creates a new array of length 5
    """
    # Check if a valid output array is provided
    if output is not None:
        out = output
        # Raise an error if the output array has the wrong length
        if len(output) != out_len:
            raise QuickFilterError("Given output array is the wrong length")
    else:
        # Create a new zero-initialized array if no output array is provided
        out = np.zeros(out_len)
    
    return out

def quick_filter(arr, window_size, idx=None, percent=0.5, output=None, edge_mode='constant', truncate_mode='same', cval=0.0):
    """
    Filters a 1D signal using the nth element in a sorted sliding window.
    
    The function slides a window of a specified size along the input signal, sorts the values
    within the window, and selects an element based on either a percentile or a specified index.
    This allows for different filtering behaviors such as min filtering (idx=0), 
    max filtering (idx=window_size-1), or median filtering (idx=window_size//2).

    Parameters:
    ----------
    arr : numpy.ndarray
        The input 1D signal to be filtered.
    window_size : int
        The size of the sliding window. Must be smaller than or equal to the length of the signal.
    idx : int, optional
        The index of the element to select from the sorted window, with 0 being the minimum and
        window_size-1 being the maximum. If not provided, the `percent` parameter will be used to
        determine the index. If `idx` is provided, it overrides `percent`.
    percent : float, optional
        The percentile to select from the sorted window (between 0.0 and 1.0). By default, the 50th 
        percentile (median) is selected. This parameter is ignored if `idx` is provided.
    output : numpy.ndarray or None, optional
        An array in which to store the filtered signal. If `None`, a new array is created.
    edge_mode : str, optional
        The mode for handling the edges of the signal. Can be one of:
        - 'nearest': Extends the edges with the nearest value.
        - 'reflect': Reflects the edge values.
        - 'mirror': Mirrors the edge values.
        - 'constant': Pads the edges with a constant value (`cval`).
        - 'wrap': Wraps the signal around the edges.
        Default is 'constant'.
    truncate_mode : str, optional
        The mode for truncating the output signal. Can be one of:
        - 'same': The output signal has the same length as the input.
        - 'valid': The output signal excludes the edge regions where the window would be incomplete.
        - 'full': The output signal includes all edge regions, extending the output length.
        Default is 'same'.
    cval : float, optional
        The constant value to use for padding if `edge_mode='constant'`. Default is 0.0.
    axes : int, optional
        The axis along which the filtering is applied (not used in this 1D implementation).
        Default is 0.

    Returns:
    -------
    numpy.ndarray
        The filtered signal, with its length depending on the `truncate_mode`.

    Raises:
    -------
    QuickFilterError
        If the input length is smaller than the window size, or if invalid parameters are provided.

    Example:
    --------
    arr = np.array([3, 1, 2, 4, 5])
    filtered = select_filter(arr, window_size=3, percent=0.5)  # Performs a median filtering
    print(filtered)  # Outputs a filtered signal
    """
    n = len(arr)

    # Validate input lengths and modes
    if n < window_size:
        raise QuickFilterError("Input length cannot be smaller than the window size")
    
    if edge_mode not in ['nearest', 'reflect', 'constant', 'mirror', 'wrap']:
        raise QuickFilterError(f"Got invalid edge-handling mode: {edge_mode}")

    if idx is not None:
        percent = idx / window_size

    if percent < 0.0 or percent > 1.0:
        raise QuickFilterError("Selection index cannot be negative or greater than the window size")
    
    qt = QuickTree()              # Maintains sorted order of window elements

    match truncate_mode:
        case 'valid':
            # Output has length reduced by window_size
            out_len = n - window_size
            out = make_output_array(output, out_len)

            for idx in range(len(arr)):
                if idx >= window_size:
                    out[idx - window_size] = qt.select(percent=percent)
                    qt.remove(arr[idx - window_size])
                qt.add(arr[idx])

            return out

        case 'same':
            # Output has the same length as input
            arr = add_edges(arr, window_size, edge_mode, cval=cval)
            out_len = n
            out = make_output_array(output, out_len)

            for idx in range(len(arr)):
                if idx >= window_size:
                    out[idx - window_size] = qt.select(percent=percent)
                    qt.remove(arr[idx - window_size])
                qt.add(arr[idx])

            return out
    
        case 'full':
            # Output has length extended by window_size
            arr = add_edges(arr, window_size, edge_mode, cval=cval)
            out_len = n + window_size - 1  # Total output length in 'full' mode
            out = make_output_array(output, out_len)

            # TODO: Fix this

            return out

        case _:
            raise QuickFilterError(f"Got invalid truncation mode: {truncate_mode}")