# --------------------------------------------------------------------------------
# Note that different from its original code https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py,
# we do not use "is_first" and "is_fast", because an issue exists otherwise for evaluating self-BLEU over training: Only
# in the first time of evaluation that the reference and hypothesis come from the same “test data” (i.e. the whole set
# of generated sentences). After that, the hypothesis keeps updated but the reference remains unchanged (due to
# “self.is_first=False”), which means hypothesis and reference are not from the same “test data” any more, and thus the
# scores obtained under that implementation is not self-BLEU scores.
# --------------------------------------------------------------------------------

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from utils.metrics.Metrics import Metrics


class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3, name='SelfBleu', portion=1):
        super().__init__()
        self.name = name
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 200  # SelfBLEU scores remain nearly unchanged for self.sample_size >= 200
        self.portion = portion  # how many posrtions to use in the evaluation, default to use the whole test dataset

    def get_name(self):
        return self.name

    def get_score(self, is_fast=False, ignore=False):
        if ignore:
            return 0

        return self.get_bleu()

    def get_reference(self):
        reference = list()
        with open(self.test_data) as real_data:
            for text in real_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
        len_ref = len(reference)

        return reference[:int(self.portion*len_ref)]

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

