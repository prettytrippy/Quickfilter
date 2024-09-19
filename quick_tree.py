from sortedcontainers import SortedList

class QuickTree:
    """
    A wrapper class for SortedList with additional logic for efficient
    element insertion, removal, and percentile-based selection.

    Attributes:
    ----------
    sl : sortedcontainers.SortedList
        The underlying sorted list that maintains the elements in sorted order.
    """
    
    def __init__(self):
        """
        Initializes an empty QuickTree instance.
        
        Uses the `SortedList` from the `sortedcontainers` library to manage
        a dynamically sorted collection of elements.
        """
        self.sl = SortedList()

    def __len__(self):
        """
        Returns the number of elements in the sorted list.
        
        Returns:
        -------
        int
            The number of elements in the QuickTree.
        """
        return len(self.sl)
    
    def add(self, elem):
        """
        Adds an element to the sorted list.
        
        The `SortedList` automatically places the element in the appropriate
        position to maintain the sorted order.
        
        Parameters:
        ----------
        elem : Any
            The element to add to the sorted list.
        """
        self.sl.add(elem)

    def remove(self, elem):
        """
        Removes an element from the sorted list if it exists.
        
        Parameters:
        ----------
        elem : Any
            The element to remove from the sorted list.
        """
        self.sl.discard(elem)
    
    def select(self, percent=0.5):
        """
        Selects and returns the element at the specified percentile of the sorted list.
        
        The percentile is converted into an index, and the element at that index
        is returned. If the index exceeds the list bounds, the highest index is used.
        
        Parameters:
        ----------
        percent : float, optional
            The percentile (0.0 to 1.0) to select the element from, by default 0.5 (median).
        
        Returns:
        -------
        Any
            The element at the specified percentile of the sorted list.
        
        Raises:
        -------
        ValueError:
            If the percentile is outside the range 0 to 1.
        
        Example:
        --------
        qt = QuickTree()
        qt.add(1)
        qt.add(2)
        qt.add(3)
        qt.select(0.5)  # Returns the median element, 2
        """
        length = len(self)
        if percent < 0.0 or percent > 1.0:
            raise ValueError("Percent must be between 0 and 1.")
        
        # Convert the percentile into an index
        idx = int(length * percent)

        if idx >= length:
            idx = length - 1
        
        return self.sl[idx]
    
    def __repr__(self):
        """
        Returns the string representation of the sorted list.
        
        Returns:
        -------
        str
            A string showing the elements in the sorted list.
        """
        return f"{self.sl}"
