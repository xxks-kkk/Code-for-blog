# trainer.py

import sys
from nerdata import *
from utils import *
from models import *


class BadNerModel():
    def __init__(self, words_to_tag_counters):
        self.words_to_tag_counters = words_to_tag_counters

    # Outputs the most common tag for every token, or "O" if
    # it hasn't been seen before.
    def decode(self, sentence):
        pred_tags = []
        for tok in sentence.tokens:
            if self.words_to_tag_counters.has_key(tok.word):
                pred_tags.append(self.words_to_tag_counters[tok.word].argmax())
            else:
                pred_tags.append("O")
        #print "pred_tags: " + repr(pred_tags)
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
    # ENG
    train = read_data("data/eng.train")
    dev = read_data("data/eng.testa")
    # DEU
    deu_train = read_data("data/deu.train")
    deu_dev = read_data("data/deu.testa")
    # Here's a few sentences...
    #print "Examples of sentences:"
    #print str(dev[1])
    #print str(dev[3])
    #print str(dev[5])
    if len(sys.argv) < 4:
        raise Exception("Synoposis: [BAD|HMM|CRF] [ENG|DEU] [BEAM|VITERBI]")
    if len(sys.argv) >= 4:
        system_to_run = sys.argv[1]     # CRF / HMM
        language_to_use = sys.argv[2]   # ENG / DEU
        decode_method = sys.argv[3]     # VITERBI / BEAM
    else:
        system_to_run = "BAD"
    # Set to True when you're ready to run your CRF on the test set to produce the final output
    run_on_test = True
    print system_to_run + " " + language_to_use + " " + decode_method
    # Train our model
    if system_to_run == "BAD":
        bad_model = train_bad_ner_model(train)
        dev_decoded = [bad_model.decode(test_ex) for test_ex in dev]
    elif system_to_run == "HMM":
        if language_to_use == "ENG":        
            hmm_model = train_hmm_model(train)
            if decode_method == "VITERBI":
                dev_decoded = [hmm_model.decode(test_ex) for test_ex in dev]
                print_evaluation(dev, dev_decoded)
            elif decode_method == "BEAM":
                dev_decoded = [hmm_model.beam_search(test_ex) for test_ex in dev]   
                print_evaluation(dev, dev_decoded)
            else:
                raise Exception("Synoposis: [BAD|HMM|CRF] [ENG|DEU] [BEAM|VITERBI] \
                                 3rd input argument is not correct: BEAM or VITERBI")
        elif language_to_use == "DEU":
            hmm_model = train_hmm_model(deu_train)
            if decode_method == "VITERBI":
                dev_decoded = [hmm_model.decode(test_ex) for test_ex in deu_dev]
                print_evaluation(deu_dev, dev_decoded)                
            elif decode_method == "BEAM":
                dev_decoded = [hmm_model.beam_search(test_ex) for test_ex in deu_dev] 
                print_evaluation(deu_dev, dev_decoded)
            else:
                raise Exception("Synoposis: [BAD|HMM|CRF] [ENG|DEU] [BEAM|VITERBI] \
                                 3rd input argument is not correct: BEAM or VITERBI")
        else:
            raise Exception("Synoposis: [BAD|HMM|CRF] [ENG|DEU] [BEAM|VITERBI] \
                             2nd input argument is not correct: ENG or DEU")
    elif system_to_run == "CRF":
        if language_to_use == "ENG":
            crf_model = train_crf_model(train)
            if decode_method == "VITERBI":
                dev_decoded = [crf_model.decode(test_ex) for test_ex in dev]
                if run_on_test:
                    test = read_data("data/eng.testb.blind")
                    test_decoded = [crf_model.decode(test_ex) for test_ex in test]
                    print_output(test_decoded, "eng.testb.out")
                # Print the evaluation statistics
                print_evaluation(dev, dev_decoded)
            elif decode_method == "BEAM":
                dev_decoded = [crf_model.beam_search(test_ex) for test_ex in dev]
                if run_on_test:
                    test = read_data("data/eng.testb.blind")
                    test_decoded = [crf_model.beam_search(test_ex) for test_ex in test]
                    print_output(test_decoded, "eng.testb.out.beam")
            else:
                raise Exception("Synoposis: [BAD|HMM|CRF] [ENG|DEU] [BEAM|VITERBI] \
                                 3rd input argument is not correct: BEAM or VITERBI")
        elif language_to_use == "DEU":
            crf_model = train_crf_model(deu_train)
            if decode_method == "VITERBI":
                dev_decoded = [crf_model.decode(test_ex) for test_ex in deu_dev]
                if run_on_test:
                    test = read_data("data/deu.testb")
                    test_decoded = [crf_model.decode(test_ex) for test_ex in test]
                    print_output(test_decoded, "deu.testb.out")
                # Print the evaluation statistics
                print_evaluation(deu_dev, dev_decoded)
            elif decode_method == "BEAM":
                dev_decoded = [crf_model.beam_search(test_ex) for test_ex in deu_dev]
                if run_on_test:
                    test = read_data("data/deu.testb")
                    test_decoded = [crf_model.beam_search(test_ex) for test_ex in test]
                    print_output(test_decoded, "deu.testb.out.beam")
            else:
                raise Exception("Synoposis: [BAD|HMM|CRF] [ENG|DEU] [BEAM|VITERBI] \
                                 3rd input argument is not correct: BEAM or VITERBI")            
    else:
        raise Exception("Synoposis: [BAD|HMM|CRF] [ENG|DEU] [BEAM|VITERBI]")

    
