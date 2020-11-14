"""
Name: Sneha Sarkar
This project focuses on teaching the machine knowledge about certain topics that people in the real world
review/tweet about. Each tweet/review has a sentiment and topic attached to it, be it positive or negative (sentiments),
about elections, oscars or Super Bowl (topics). This project focuses on categorizing tweets/reviews after being trained
from a given dataset of such categories. This effectively sorts the tweets/reviews to cater to certain interest areas
of different people around the world. Furthermore, this project uses a Tries data structure to store such information,
which allows fast and time efficient access, add, remove and search operations of each data.
"""

from __future__ import annotations
from typing import Tuple, Dict, List


class TrieNode:
    """
    Implementation of a trie node.
    """

    # DO NOT MODIFY

    __slots__ = "children", "is_end"

    def __init__(self, arr_size: int = 26) -> None:
        """
        Constructs a TrieNode with arr_size slots for child nodes.
        :param arr_size: Number of slots to allocate for child nodes.
        :return: None
        """
        self.children = [None] * arr_size
        self.is_end = 0

    def __str__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        if self.empty():
            return "..."
        children = self.children  # to shorten proceeding line
        return str({chr(i + ord("a")) + "*"*min(children[i].is_end, 1): children[i] for i in range(26) if children[i]})

    def __repr__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        return self.__str__()

    def __eq__(self, other: TrieNode) -> bool:
        """
        Compares two TrieNodes for equality.
        :return: True if two TrieNodes are equal, else False
        """
        if not other or self.is_end != other.is_end:
            return False
        return self.children == other.children

    # Implement Below

    def empty(self) -> bool:
        """
        Returns True if TrieNode is leaf (has no children).
        :return: Bool.
        """
        if self.children == [None]*26:
            return True
        else:
            return False

    @staticmethod
    def _get_index(char: str) -> int:
        """
        Returns the integer index of a character
        in a-z or A-Z.
        :param char: character to be mapped to integer
        :return: integer index of a character in a-z or A-Z
        """
        return ord(char.lower()) - 97

    def get_child(self, char: str) -> TrieNode:
        """
        Retrieves and returns the child TrieNode
        at the index returned by _get_index(char)
        :param char: character of child TrieNode to retrieve
        :return: child TrieNode
        at the index returned by _get_index(char)
        """
        if self.empty():
            return None
        else:
            return self.children[self._get_index(char)]

    def set_child(self, char: str) -> None:
        """
        Creates TrieNode and stores it in children
        at the index returned by _get_index(char)
        :param char: character of child TrieNode to create
        :return: None
        """
        node = TrieNode()
        self.children[self._get_index(char)] = node

    def delete_child(self, char: str) -> None:
        """
        Deletes the child TrieNode at the index returned
        by _get_index(char) by setting it to None
        :param char: character of child TrieNode to delete
        :return: None
        """
        if not self.empty():
            self.children[self._get_index(char)] = None


class Trie:
    """
    Implementation of a trie.
    """

    # DO NOT MODIFY

    __slots__ = "root", "unique", "size"

    def __init__(self) -> None:
        """
        Constructs an empty Trie.
        :return: None.
        """
        self.root = TrieNode()
        self.unique = 0
        self.size = 0

    def __str__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return "Trie Visual:\n" + str(self.root)

    def __repr__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return self.__str__()

    def __eq__(self, other: Trie) -> bool:
        """
        Compares two Tries for equality.
        :return: True if two Tries are equal, else False
        """
        return self.root == other.root

    # Implement Below

    def add(self, word: str) -> int:
        """
        RECURSIVE
        Adds word to Trie by traversing the Trie
        from the root downward and creating TrieNodes as necessary
        :param word: String to be added to the Trie.
        :return: number of times word exists in the Trie
        """

        def add_inner(node: TrieNode, index: int) -> int:
            if not node.get_child(word[index]) and index == len(word) - 1:
                node.set_child(word[index])
                node_ = node.get_child(word[index])
                node_.is_end += 1
                return node_.is_end
            elif node.get_child(word[index]) and index == len(word) - 1:
                node.get_child(word[index]).is_end += 1
                return node.get_child(word[index]).is_end
            elif not node.get_child(word[index]):
                node.set_child(word[index])
                node_ = node.get_child(word[index])
                return add_inner(node_, index + 1)
            else:
                node_ = node.get_child(word[index])
                return add_inner(node_, index + 1)

        ret = add_inner(self.root, 0)
        if ret == 1:
            self.unique += 1
        self.size += 1
        return ret

    def search(self, word: str) -> int:
        """
        RECURSIVE
        Searches word in Trie by traversing the Trie
        from the root downward  until the last character
        of word is reached.

        :param word: String to be searched for in the Trie
        :return: 0 if word is not found in Trie,
        else number of times word exists in the Trie
        """
        def search_inner(node: TrieNode, index: int) -> int:
            if not node.get_child(word[index]) and index == len(word) - 1:
                return 0
            elif node.get_child(word[index]) and index == len(word) - 1:
                return node.get_child(word[index]).is_end
            elif not node.get_child(word[index]):
                return 0
            else:
                node_ = node.get_child(word[index])
                return search_inner(node_, index + 1)

        ret = search_inner(self.root, 0)
        return ret


    def delete(self, word: str) -> int:
        """
        RECURSIVE
        Deletes word from the Trie
        :param word: String to be deleted from the Trie.
        :return: 0 if word is not found in Trie, else
        number of times word existed in the Trie before deletion
        """
        def delete_inner(node: TrieNode, index: int) -> Tuple[int, bool]:
            if not node.get_child(word[index]) and index == len(word) - 1:
                return 0, False
            elif not node.get_child(word[index]):
                return 0, False
            elif node.get_child(word[index]) and index == len(word) - 1:
                end = node.get_child(word[index]).is_end
                if node.get_child(word[index]).empty():
                    node.get_child(word[index]).is_end = 0
                    return end, True
                node.get_child(word[index]).is_end = 0
                return end, False
            else:
                node_ = node.get_child(word[index])
                ret_ = delete_inner(node_, index + 1)
                if node_.get_child(word[index+1]):
                    if node_.get_child(word[index+1]).is_end == 0 and ret_[1]:
                        node.get_child(word[index]).delete_child(word[index+1])
                        if node.get_child(word[index]).is_end == 0 and node.get_child(word[index]).empty():
                            return ret_[0], True
                return ret_[0], False

        ret = delete_inner(self.root, 0)

        if ret[1]:
            self.root.delete_child(word[0])

        if ret[0] == 0:
            return ret[0]

        self.unique -= 1
        self.size = self.size - ret[0]
        return ret[0]

    def __len__(self) -> int:
        """
        Returns the total number of words
        (including repetitions) in the vocabulary
        :return: total number of words in Trie.
        """
        return self.size

    def __contains__(self, word: str) -> bool:
        """
        Uses search(self,word) to check if a word exists in Trie
        :param word: word to search for
        :return: bool
        """
        count = self.search(word)
        if count == 0:
            return False
        else:
            return True

    def empty(self) -> bool:
        """
        Returns True if vocabulary of Trie is empty, else False.
        :return: bool
        """
        if self.size > 0:
            return False
        else:
            return True

    def get_vocabulary(self, prefix: str = "") -> Dict[str, int]:
        """
        Returns a dictionary of (word, count) pairs containing
        every word in the Trie beginning with prefix.
        :param prefix: Prefix string to match with words in Trie.
        :return: dictionary of of (word, count) pairs
        """
        dict_ = {}

        def get_vocabulary_inner(node, suffix):
            if node.empty():
                dict_[suffix] = node.is_end
            elif node.is_end > 0:
                i_ = 0
                dict_[suffix] = node.is_end
                for child in node.children:
                    if child:
                        temp = suffix
                        suffix = suffix + chr(i_ + 97)
                        get_vocabulary_inner(child, suffix)
                        suffix = temp
                    i_ += 1
            else:
                i_ = 0
                for child in node.children:
                    if child:
                        temp = suffix
                        suffix = suffix + chr(i_ + 97)
                        get_vocabulary_inner(child, suffix)
                        suffix = temp
                    i_ += 1

        i = 0
        root_ = self.root
        while i < len(prefix):
            root_ = root_.get_child(prefix[i])
            if not root_:
                return dict_
            i += 1

        if self.empty():
            return dict_
        if prefix == "":
            get_vocabulary_inner(root_, "")
        else:
            get_vocabulary_inner(root_, prefix)
        return dict_

    def autocomplete(self, word: str) -> Dict[str, int]:
        """
        Returns a dictionary of (word, count) pairs containing every
        word in the Trie which matches the template of word, where
        periods (.) in word may be filled with any character.
        :param word: Template string to match with words in Trie.
        :return: dictionary of (word, count) pairs
        """
        dict_ = {}

        def autocomplete_inner(node, prefix, index):
            if node and index == len(word) - 1 and word[index] == ".":
                i_ = 0
                for child in node.children:
                    if child:
                        temp = prefix
                        prefix = prefix + chr(i_ + 97)
                        if child.is_end > 0:
                            dict_[prefix] = child.is_end
                        prefix = temp
                    i_ += 1
            elif node and word[index] == ".":
                i_ = 0
                for child in node.children:
                    if child:
                        temp = prefix
                        prefix = prefix + chr(i_ + 97)
                        autocomplete_inner(child, prefix, index+1)
                        prefix = temp
                    i_ += 1
            elif node and index == len(word) - 1 and word[index] != ".":
                node_ = node.get_child(word[index])
                if node_:
                    if node_.is_end > 0:
                        prefix = prefix + word[index]
                        dict_[prefix] = node_.is_end
            elif node:
                node = node.get_child(word[index])
                prefix = prefix + word[index]
                autocomplete_inner(node, prefix, index+1)

        if self.empty():
            return dict_
        if word[0] == ".":
            autocomplete_inner(self.root, "", 0)
        else:
            autocomplete_inner(self.root, "", 0)
        return dict_


class TrieClassifier:
    """
    Implementation of a trie-based text classifier.
    """

    # DO NOT MODIFY

    __slots__ = "tries"

    def __init__(self, classes: List[str]) -> None:
        """
        Constructs a TrieClassifier with specified classes.
        :param classes: List of possible class labels of training and testing data.
        :return: None.
        """
        self.tries = {}
        for cls in classes:
            self.tries[cls] = Trie()

    @staticmethod
    def accuracy(labels: List[str], predictions: List[str]) -> float:
        """
        Computes the proportion of predictions that match labels.
        :param labels: List of strings corresponding to correct class labels.
        :param predictions: List of strings corresponding to predicted class labels.
        :return: Float proportion of correct labels.
        """
        correct = sum([1 if label == prediction else 0 for label, prediction in zip(labels, predictions)])
        return correct / len(labels)

    # Implement Below

    def fit(self, class_strings: Dict[str, List[str]]) -> None:
        """
        Adds every individual word in the list of strings
        associated with each class to the Trie corresponding
        to the class in self.tries
        :param class_strings: A dictionary of (class, List[str])
        pairs to train the classifier on.
        :return: None
        """
        for key, val in class_strings.items():
            for tweet in val:
                line = tweet.split()
                for word in line:
                    self.tries[key].add(word)

    def predict(self, strings: List[str]) -> List[str]:
        """
        Returns a list of predicted classes corresponding
        to the input strings.
        :param strings:A list of strings (tweets) to be classified.
        :return: list of predicted classes
        """
        dict_ = {}
        list_ = []
        for key, val in self.tries.items():
            dict_[key] = 0

        for tweet in strings:
            line = tweet.split()
            for word in line:
                for key, val in self.tries.items():
                    node_n = val.search(word)
                    dict_[key] = dict_[key] + node_n
            for key, val in dict_.items():
                dict_[key] = val / len(self.tries[key])
            max_ = 0
            for key, val in dict_.items():
                if val > max_:
                    max_ = val
                    max_class = key

            list_.append(max_class)
        return list_



