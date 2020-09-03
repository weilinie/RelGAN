import random

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from utils.metrics.Metrics import Metrics


class Bleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, name='Bleu', portion=1):
        super().__init__()
        self.name = name
        self.test_data = test_text
        self.real_data = real_text
        self.gram = gram
        self.sample_size = 100  # BLEU scores remain nearly unchanged for self.sample_size >= 100
        self.reference = None
        self.is_first = True
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def get_name(self):
        return self.name

    def get_score(self, is_fast=False, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        return self.get_bleu()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)

            # randomly choose a portion of test data
            # In-place shuffle
            random.shuffle(reference)
            len_ref = len(reference)
            reference = reference[:int(self.portion*len_ref)]

            self.reference = reference

            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            i = 0
            for hypothesis in test_data:
                if i >= self.sample_size:
                    break
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(self.calc_bleu(reference, hypothesis, weight))
                i += 1
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)
