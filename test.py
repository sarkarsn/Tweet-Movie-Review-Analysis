import unittest

import string
from Trie import Trie, TrieNode, TrieClassifier


class TestProject(unittest.TestCase):

    def test_trie_classifier_artificial(self):
        """
        Trains the machine to learn terms relating to any set of
        topics/categories and then tests to predict which topic/categories
        the string belongs to.
        """
      
        # binary classifier: positive/negative-sentiment sentences
        classes = ["positive", "negative"]
        train_positive = ["sun sunny sunshine",
                          "smile smiling smiled",
                          "laugh laughing laughed",
                          "happy happier happiest"]
        train_negative = ["rain rainy rained",
                          "frown frowning frowned",
                          "cry crying cried",
                          "sad sadder saddest"]

        training_strings = {"positive": train_positive, "negative": train_negative}
        clf = TrieClassifier(classes)
        clf.fit(training_strings)


        # test single predictions
        pred = clf.predict(["the sunshine made me smile today"])
        self.assertEqual(pred, ["positive"])
        pred = clf.predict(["laughing with my best friends always makes me happier"])
        self.assertEqual(pred, ["positive"])
        pred = clf.predict(["the clouds and rain always make me sad"])
        self.assertEqual(pred, ["negative"])
        pred = clf.predict(["she frowned and cried after hearing the bad news"])
        self.assertEqual(pred, ["negative"])

        # test multiple predictions
        test_positive = ["the sunshine made me smile today",
                         "laughing with my best friends always makes me happier",
                         "when youre happy you dont frown or cry"]
        pred = clf.predict(test_positive)
        self.assertEqual(pred, ["positive", "positive", "negative"])
        truth = ["positive" for _ in test_positive]
        acc = clf.accuracy(truth, pred)
        self.assertAlmostEqual(acc, 2/3)

        test_negative = ["the clouds and rain always make me sad",
                         "she had not laughed nor smiled for days",
                         "without sunshine she found it hard to be happy"]
        pred = clf.predict(test_negative)
        self.assertEqual(pred, ["negative", "positive", "positive"])
        truth = ["negative" for _ in test_positive]
        acc = clf.accuracy(truth, pred)
        self.assertAlmostEqual(acc, 1/3)

        # multi-classifier: subject-area strings
        classes = ["computer", "business", "history", "art"]
        train_computer = ["computer computers computed computing computational",
                          "program programs programmed programming programmer",
                          "code codes coded coding coder"]
        train_business = ["present presents presented presenting presentation",
                          "negotiate negotiates negotiated negotiating negotiation",
                          "sell sells sold selling"]
        train_history = ["research researches researched researching researcher",
                         "write writes wrote writing",
                         "read reads reading"]
        train_art = ["paint paints painted painting",
                     "draw draws drew drawing",
                     "sculpt sculpts sculpted sculpting sculpture"]
        training_strings = {"computer": train_computer, "business": train_business,
                              "history": train_history, "art": train_art}
        clf = TrieClassifier(classes)
        clf.fit(training_strings)

        test_all = ["she enjoys writing computer programs",
                    "the ceo presented an amazing product and sold the team on her idea",
                    "she aspires to be a history professor who writes research papers on ancient greece",
                    "she drew a sketch of the sculpture she envisioned before sculpting it",
                    "as a computer science researcher she reads and writes code all day",
                    "she negotiated a deal with the museum to display her latest masterpiece at the new exhibit",
                    "the business college established a new program in supply chain last summer"]
        pred = clf.predict(test_all)
        self.assertEqual(pred, ["computer", "business", "history", "art", "history", "business", "computer"])
        truth = ["computer", "business", "history", "art", "computer", "art", "business"]
        acc = clf.accuracy(truth, pred)
        self.assertAlmostEqual(acc, 4/7)

    def test_trie_classifier_real(self):
        """
        Trains the machine using a large dataset of various positive and
        negative movie reviews and uses another large test dataset of movie reviews
        to predict whether the review is positive or negative.
        """

        # load training data
        classes = ["positive", "negative"]
        train_positive, train_negative = [], []
        with open("movie_reviews_train.txt") as tsv:
            lines = tsv.readlines()
            for line in lines[1:]:                          # skip header row
                sentence, sentiment = line.split("\t")      # split tab-delimited variables
                sentence = "".join([c for c in sentence.lower()
                                    if c in string.ascii_lowercase or c.isspace()])     # clean sentence and sentiment
                sentiment = sentiment.strip()
                if sentiment == "positive":
                    train_positive.append(sentence)
                else:
                    train_negative.append(sentence)

        # fit classifier
        training_strings = {"positive": train_positive, "negative": train_negative}
        clf = TrieClassifier(classes)
        clf.fit(training_strings)

        # load testing data
        test_strings, truth = [], []
        with open("movie_reviews_test.txt") as tsv:
            lines = tsv.readlines()
            for line in lines[1:]:                      # skip header row
                sentence, sentiment = line.split("\t")  # split tab-delimited variables
                sentence = "".join([c for c in sentence.lower()
                                    if c in string.ascii_lowercase or c.isspace()])     # clean sentence and sentiment
                sentiment = sentiment.strip()
                test_strings.append(sentence)
                truth.append(sentiment)

        # predict and validate
        pred = clf.predict(test_strings)
        print("predicted sentiments list from test dataset:\n",pred)
        print("actual sentiments list from test dataset:\n",truth)

        acc = clf.accuracy(truth, pred)
        print("Accuracy of predictions: ",acc)


if __name__ == '__main__':
    unittest.main()



