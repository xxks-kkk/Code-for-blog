# models.py

from nerdata import *
from utils import *

import numpy as np


# Scoring function for sequence models based on conditional probabilities.
# Scores are provided for three potentials in the model: initial scores (applied to the first tag),
# emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
class ProbabilisticSequenceScorer(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence.tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    def decode(self, sentence):
        raise Exception("IMPLEMENT ME")


# Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
# Any word that only appears once in the corpus is replaced with UNK. A small amount
# of additive smoothing is applied to
def train_hmm_model(sentences):
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter.increment_count(token.word, 1.0)
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in xrange(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.get_index(bio_tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print repr(init_counts)
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print "Tag indexer: " + repr(tag_indexer)
    print "Initial state log probabilities: " + repr(init_counts)
    print "Transition log probabilities: " + repr(transition_counts)
    print "Emission log probs too big to print..."
    print "Emission log probs for India: " + repr(emission_counts[:,word_indexer.get_index("India")])
    print "Emission log probs for Phil: " + repr(emission_counts[:,word_indexer.get_index("Phil")])
    print "   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)"
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


# Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
# At test time, unknown words will be replaced by UNKs.
def get_word_index(word_indexer, word_counter, word):
    if word_counter.get_count(word) < 1.5:
        return word_indexer.get_index("UNK")
    else:
        return word_indexer.get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    def decode(self, sentence):
        raise Exception("IMPLEMENT ME")


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    print "Extracting features"
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentences[i]))] for i in xrange(0, len(sentences))]
    for sentence_idx in xrange(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print "Ex " + repr(sentence_idx) + "/" + repr(len(sentences))
        for word_idx in xrange(0, len(sentences[sentence_idx])):
            for tag_idx in xrange(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    raise Exception("IMPLEMENT THE REST OF ME")


# Extracts emission features for tagging the word at word_index with tag.
# add_to_indexer is a boolean variable indicating whether we should be expanding the indexer or not:
# this should be True at train time (since we want to learn weights for all features) and False at
# test time (to avoid creating any features we don't have weights for).
def extract_emission_features(sentence, word_index, tag, feature_indexer, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in xrange(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in xrange(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)
