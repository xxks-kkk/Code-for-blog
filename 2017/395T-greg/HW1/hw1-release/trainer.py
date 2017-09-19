# trainer.py

import sys
from nerdata import *
from utils import *
from models import *


class BadNerModel():
    def __init__(self, words_to_tag_counters):
        self.words_to_tag_counters = words_to_tag_counters

    def decode(self, sentence):
        pred_tags = []
        for tok in sentence.tokens:
            if self.words_to_tag_counters.has_key(tok.word):
                pred_tags.append(self.words_to_tag_counters[tok.word].argmax())
            else:
                pred_tags.append("O")
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


def train_bad_ner_model(training_set):
    words_to_tag_counters = {}
    for sentence in training_set:
        tags = sentence.get_bio_tags()
        for idx in xrange(0, len(sentence)):
            word = sentence.tokens[idx].word
            if not words_to_tag_counters.has_key(word):
                words_to_tag_counters[word] = Counter()
                words_to_tag_counters[word].increment_count(tags[idx], 1.0)
    return BadNerModel(words_to_tag_counters)


if __name__ == '__main__':
    # Load the training and test data
    train = read_data("data/eng.train")
    dev = read_data("data/eng.testa")
    # Here's a few sentences...
    print "Examples of sentences:"
    print str(dev[1])
    print str(dev[3])
    print str(dev[5])
    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "BAD"
    # Set to True when you're ready to run your CRF on the test set to produce the final output
    run_on_test = True
    # Train our model
    if system_to_run == "BAD":
        bad_model = train_bad_ner_model(train)
        dev_decoded = [bad_model.decode(test_ex) for test_ex in dev]
    elif system_to_run == "HMM":
        hmm_model = train_hmm_model(train)
        dev_decoded = [hmm_model.decode(test_ex) for test_ex in dev]
    elif system_to_run == "CRF":
        crf_model = train_crf_model(train)
        dev_decoded = [crf_model.decode(test_ex) for test_ex in dev]
        if run_on_test:
            test = read_data("data/eng.testb.blind")
            test_decoded = [crf_model.decode(test_ex) for test_ex in test]
            print_output(test_decoded, "eng.testb.out")
    else:
        raise Exception("Pass in either BAD, HMM, or CRF to run the appropriate system")
    # Print the evaluation statistics
    print_evaluation(dev, dev_decoded)
