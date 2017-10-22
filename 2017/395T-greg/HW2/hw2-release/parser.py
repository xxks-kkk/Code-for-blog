# parser.py

import sys
from models import *
from random import shuffle


if __name__ == '__main__':
    # Load the training and test data
    print "Reading train data..."
    train = read_data("data/train.conllx")
    shuffle(train)
    print "Kept " + repr(len(train)) + " exs"
    print "Reading dev data..."
    dev = read_data("data/dev.conllx")
    # Here's a few sentences...
    print "Examples of sentences:"
    print str(dev[1])
    print str(dev[3])
    print str(dev[5])

    # Set to true to produce final output
    run_on_test = False
    parsed_dev = []
    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "TEST_TRANSITIONS"
    if system_to_run == "TEST_TRANSITIONS":
        for idx in xrange(0, len(dev)):
            parsed_sentence = dev[idx]
            print "INDEX: " + repr(idx)
            (decisions, states) = get_decision_sequence(parsed_sentence)
            parsed_dev.append(ParsedSentence(parsed_sentence.tokens, states[-1].get_dep_objs(len(parsed_sentence))))
    elif system_to_run == "GREEDY":
        trained_model = train_greedy_model(train)
        print "Parsing dev"
        parsed_dev = [trained_model.parse(sent) for sent in dev]
        if run_on_test:
            print "Parsing test"
            test = read_data("data/test.conllx.blind")
            test_decoded = [trained_model.parse(test_ex) for test_ex in test]
            print_output(test_decoded, "test.conllx.out")
    elif system_to_run == "BEAM":
        trained_model = train_beamed_model(train)
        print "Parsing dev"
        parsed_dev = [trained_model.parse(sent) for sent in dev]
        if run_on_test:
            print "Parsing test"
            test = read_data("data/test.conllx.blind")
            test_decoded = [trained_model.parse(test_ex) for test_ex in test]
            print_output(test_decoded, "test.conllx.out")
    else:
        raise Exception("Pass in either TEST_TRANSITIONS, GREEDY, or BEAM to run the appropriate system")
    print_evaluation(dev, parsed_dev)
