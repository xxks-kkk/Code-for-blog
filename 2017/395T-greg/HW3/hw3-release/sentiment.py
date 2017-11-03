# parser.py

import sys
from models import *
from sentiment_data import *


if __name__ == '__main__':
    # Use either 50-dim or 300-dim vectors
    #word_vectors = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    word_vectors = read_word_embeddings("data/glove.6B.300d-relativized.txt")

    # Load train, dev, and test exs
    train_exs = read_and_index_sentiment_examples("data/train.txt", word_vectors.word_indexer)
    dev_exs = read_and_index_sentiment_examples("data/dev.txt", word_vectors.word_indexer)
    test_exs = read_and_index_sentiment_examples("data/test-blind.txt", word_vectors.word_indexer)
    print repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples"

    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "CNN"
    if system_to_run == "FF":
        test_exs_predicted = train_ffnn(train_exs, dev_exs, test_exs, word_vectors)
        # Write the test set output
        write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)
    if system_to_run == "FF2":
        test_exs_predicted = train_ffnn2(train_exs, dev_exs, test_exs, word_vectors)
    elif system_to_run == "CNN":
        test_exs_predicted = train_cnn(train_exs, dev_exs, test_exs, word_vectors)
        # Write the test set output
        write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)
    elif system_to_run == "LSTM":
        test_exs_predicted = train_lstm(train_exs, dev_exs, test_exs, word_vectors)
    else:
        raise Exception("Pass in either FF, FF2, CNN, LSTM to run the appropriate system")