import numpy as np


class SumTree:
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """

    def __init__(self, capacity: int):
        # Initialize the sum tree with the given capacity
        self.capacity = capacity  # Maximum number of elements in the tree
        self.tree = np.zeros(2 * capacity - 1)  # Parent nodes + leaf nodes
        self.data = np.zeros(capacity, dtype=object)  # Array to store actual data
        self.n_entries = 0  # Counter for the number of entries added
        self.data_pointer = 0  # Pointer to the next leaf index for new data

    def add(self, p: float, data: object):
        # Add priority and data to the tree
        tree_idx = self.data_pointer + self.capacity - 1  # Index in the tree array
        self.data[self.data_pointer] = data  # Insert the data into the data array
        self.update(tree_idx, p)  # Update the tree with the new priority

        # Move to the next data index and wrap around if necessary
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        # Increment the number of entries as long as we haven't filled the tree
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx: int, p: float):
        # Update the tree with the new priority score
        change = p - self.tree[tree_idx]  # Change in priority
        self.tree[tree_idx] = p  # Set the new priority
        # Propagate the change through the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # Move up to the parent node
            self.tree[tree_idx] += change  # Update the parent node's priority

    def get_leaf(self, v: float):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        # Retrieve the leaf node for the given value
        parent_idx = 0  # Start from the root
        while True:
            cl_idx = 2 * parent_idx + 1  # Left child index
            cr_idx = cl_idx + 1  # Right child index
            # If we reach bottom, end the search
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # Move to the next child node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # Calculate the index in the data array
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        # Get the total priority score
        return self.tree[0]  # The root contains the total priority
