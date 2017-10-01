# models.py

from nerdata import *
from utils import *

import numpy as np
from sys import maxint
import sys
import time
import os
from scipy.misc import logsumexp


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

    # Scores the initial HMM state
    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    # Scores an HMM state transition
    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    # Scores the HMM emission
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
        pred_tags = []     
        scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)
 
        # Implements the viterbi algorithm based upon fig 10.8 of the book "Speech and Language Processing (3rd ed. draft)"
        # NOTE: since the whole code adopts the $\pi, QA$ representation (see Section 9.2), we don't need the start and end state.
        T = len(sentence)                     # Number of observations
        N = len(self.tag_indexer)             # Number of states
        viterbi = np.zeros(shape=(N,T))       # Create a path probability matrix viterbi[N,T]
        backpointer = np.zeros(shape=(N,T))

        # Initialization step
        for s in range(N):
            # "+" because the probabilities are log-based
            viterbi[s,0] = scorer.score_init(sentence, s) + scorer.score_emission(sentence,s, 0)
            backpointer[s,0] = 0

        # Recursion step
        for t in range(1,T):
            for s in range(N):
                tmp1 = np.zeros(N) # build the candidate values for viterbi
                tmp2 = np.zeros(N) # build the candidate values for backpointer
                for s_tmp in range(N):
                    # "+" because the probabilities are log-based
                    tmp1[s_tmp] = viterbi[s_tmp,t-1] + scorer.score_transition(sentence,s_tmp,s) + scorer.score_emission(sentence,s,t)
                    tmp2[s_tmp] = viterbi[s_tmp,t-1] + scorer.score_transition(sentence,s_tmp,s)
                viterbi[s,t] = np.max(tmp1)
                backpointer[s,t] = np.argmax(tmp2)

        # Termination step (skipped because we don't have the end state)
        # Backtrace
        pred_tags.append(self.tag_indexer.get_object(np.argmax(viterbi[:,T-1])))
        for t in xrange(1,T):
            pred_tags.append(self.tag_indexer.get_object(backpointer[self.tag_indexer.get_index(pred_tags[-1]),T-t]))         

        pred_tags = list(reversed(pred_tags))
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

    def beam_search(self, sentence):      
        pred_tags = []
        T = len(sentence)
        N = len(self.tag_indexer)
        beam_size = 9
        beam_list = []

        scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)
        pred_tags = []
        T = len(sentence) 
        N = len(self.tag_indexer)
                
        # Initialization step
        beam = Beam(beam_size)
        for s in range(N):
            score = scorer.score_init(sentence, s) + scorer.score_emission(sentence,s,0)
            beam.add(self.tag_indexer.get_object(s), score)
        beam_list.append(beam)
        
        # Recursion step
        for t in range(1,T):
            beam = Beam(beam_size)
            for i in beam_list[t-1].get_elts_and_scores():
                j = self.tag_indexer.index_of(i[0])
                for s in range(N):
                    score = scorer.score_transition(sentence, j, s) + scorer.score_emission(sentence, s, t)
                    beam.add(self.tag_indexer.get_object(s), i[1] + score)
            beam_list.append(beam)

        # Backtrace
        beam_list = reversed(beam_list)
        for beam in beam_list:
            pred_tags.append(beam.head())

        pred_tags = list(reversed(pred_tags))
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


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

        # Start a timer
        #start_time = time.time()

        pred_tags = []
        T = len(sentence) 
        N = len(self.tag_indexer)
        viterbi = np.zeros(shape=(N,T))
        backpointer = np.zeros(shape=(N,T))

        score_matrix = np.zeros(shape=(N,T))
        for s in range(N):
            for t in range(T):
                features = extract_emission_features(sentence, 
                                                     t, 
                                                     self.tag_indexer.get_object(s),
                                                     self.feature_indexer,
                                                     add_to_indexer=False)
                score = sum([self.feature_weights[i] for i in features])
                score_matrix[s,t] = score
                
        # Initialization step
        for s in range(N):
            # "+" because the probabilities are log-based
            tag = str(self.tag_indexer.get_object(s))
            if (isI(tag)):
                viterbi[s,0] = -maxint
            else:  
                viterbi[s,0] = score_matrix[s,0]
            backpointer[s,0] = 0
        
        # Recursion step
        for t in range(1,T):
            for s in range(N):
                tmp1 = np.zeros(N)
                tmp2 = np.zeros(N)
                for s_tmp in range(N):
                    # "+" because the probabilities are log-based
                    # We want to ban out certain scenario:
                    # 1. We cannot have O, I tag sequence of any type
                    # 2. We cannot have I-x, I-y tag sequence of different types
                    # 3. We cannot have B-x, I-y tag sequence of any type of I other than x
                    prev_tag = str(self.tag_indexer.get_object(s_tmp))
                    curr_tag = str(self.tag_indexer.get_object(s))
                    if (isO(prev_tag) and isI(curr_tag)) or \
                       (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) or \
                       (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
                        tmp1[s_tmp] = -maxint
                        tmp2[s_tmp] = -maxint
                    else:
                        tmp1[s_tmp] = viterbi[s_tmp,t-1] + score_matrix[s,t]
                        tmp2[s_tmp] = viterbi[s_tmp,t-1]
                viterbi[s,t] = np.max(tmp1)
                backpointer[s,t] = np.argmax(tmp2)
                # Termination step (skipped because we don't have the end state)
        # Backtrace
        pred_tags.append(self.tag_indexer.get_object(np.argmax(viterbi[:,T-1])))
        for t in xrange(1,T):
            pred_tags.append(self.tag_indexer.get_object(backpointer[self.tag_indexer.get_index(pred_tags[-1]),T-t]))         

        pred_tags = list(reversed(pred_tags))

        # Calculate the amount of time used for one sentence
        # The actual time per this function call is around 1s
        #elapsed_time = time.time() - start_time
        #hours, rem = divmod(elapsed_time, 3600)
        #minutes, seconds = divmod(rem, 60)
        #print('[viterbi] time eplased: {:0>2}:{:05.2f}'.format(int(minutes), seconds))

        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

    def beam_search(self, sentence):
        # NOTE: beam is sorted by its score. The largest score will stay at top   

        # Start a timer
        #start_time = time.time()

        pred_tags = []
        T = len(sentence)
        N = len(self.tag_indexer)
        beam_size = 2
        beam_list = []

        pred_tags = []
        T = len(sentence) 
        N = len(self.tag_indexer)
                
        # Initialization step
        beam = Beam(beam_size)
        for s in range(N):
            tag = str(self.tag_indexer.get_object(s))
            if (isI(tag)):
                score = -maxint
            else:
                features = extract_emission_features(sentence,
                                                     0,
                                                     self.tag_indexer.get_object(s),
                                                     self.feature_indexer,
                                                     False)
                score = score_indexed_features(features, self.feature_weights)
            beam.add(self.tag_indexer.get_object(s),score)
        beam_list.append(beam)
        
        # Recursion step
        for t in range(1,T):
            beam = Beam(beam_size)
            for i in beam_list[t-1].get_elts_and_scores():
                j = self.tag_indexer.index_of(i[0])
                for s in range(N):
                    # We want to ban out certain scenario:
                    # 1. We cannot have O, I tag sequence of any type
                    # 2. We cannot have I-x, I-y tag sequence of different types
                    # 3. We cannot have B-x, I-y tag sequence of any type of I other than x
                    prev_tag = str(j)
                    curr_tag = str(self.tag_indexer.get_object(s))
                    if (isO(prev_tag) and isI(curr_tag)) and \
                       (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) and \
                       (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
                        score = -maxint
                    else:
                        features = extract_emission_features(sentence, 
                                                             t, 
                                                             self.tag_indexer.get_object(s),
                                                             self.feature_indexer,
                                                             add_to_indexer=False)
                        score = score_indexed_features(features, self.feature_weights)
                    beam.add(self.tag_indexer.get_object(s), i[1] + score)
            beam_list.append(beam)

        # Backtrace
        beam_list = reversed(beam_list)
        for beam in beam_list:
            pred_tags.append(beam.head())

        pred_tags = list(reversed(pred_tags))

        # Calculate the amount of time used for one sentence
        # elapsed_time = time.time() - start_time
        # hours, rem = divmod(elapsed_time, 3600)
        # minutes, seconds = divmod(rem, 60)
        # print('[beam] time eplased: {:0>2}:{:05.2f}'.format(int(minutes), seconds))

        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)

    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentences[i]))] for i in xrange(0, len(sentences))]
    # Controls the number of sentences we used for training 
    # (for development purpose; set to 1 when you are done with development)
    data_adjustment = 1
    num_sentences_use = int(len(sentences)/data_adjustment)
    for sentence_idx in xrange(0, num_sentences_use):
        #if sentence_idx % 100 == 0:
        #   print "Ex " + "sentence_idx" + "num sentences"
        #   print "Ex " + repr(sentence_idx) + "/" + repr(len(sentences))
        for word_idx in xrange(0, len(sentences[sentence_idx])):
            for tag_idx in xrange(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], 
                                                                                           word_idx, 
                                                                                           tag_indexer.get_object(tag_idx), 
                                                                                           feature_indexer, 
                                                                                           add_to_indexer=True)

    feature_weights = np.random.rand(len(feature_indexer))
    # Learn the feature weights for the NER system
    numIters = 27
    epoch = 0
    eta = 1   ## learning rate
    while True: 
        #print "epoch: " + str(epoch)
        loss = 0
        # Start a timer
        start_time = time.time()
        for sentence_idx in range(num_sentences_use):
            #print "sentence_idx: " + str(sentence_idx)
            gradients = Counter()

            T = len(sentences[sentence_idx])     # Number of observations
            N = len(tag_indexer)                 # Number of states

            # Construct feature matrix
            feature_matrix = np.zeros(shape=(N,T))
            for s in range(N):
                for t in range(T):
                    # Calculate $\phi_e(y_i,i,\pmb{x})$
                    feature_matrix[s,t] = np.sum(np.take(feature_weights, feature_cache[sentence_idx][t][s])) 


            forward = np.zeros(shape=(N,T))      # create a matrix to store the forward probabilities
            backward = np.zeros(shape=(N,T))     # create a matrix to store the backward probabilities

            #   Forward-backward algorithm to calculate P(y_i = s | X)
            #   NOTE: I ignore transition feature for now
            ##  Forward 
            ### Initialization step
            for s in range(N):
                forward[s,0] = feature_matrix[s,0]
            ### Recursion step
            for t in range(1,T):
                for s in range(N):
                    #sum = logsumexp(forward[:,t-1])
                    sum = 0
                    for i in range(N):
                        if i == 0:
                            sum = forward[i, t-1]
                        else:
                            sum = np.logaddexp(sum, forward[i,t-1])
                    forward[s,t] = feature_matrix[s,t] + sum 

            ### Termination step (skipped because we don't have the end state)

            ## Backward
            ### Initialization step
            for s in range(N):
                backward[s,T-1] = 0 # alternatively, backward[:,-1] = 0
            ### Recursion step
            for t in range(1,T):
                for s in range(N):
                    # This one line code is equivalent to the following sum calculation.
                    # However, based on the experiments, the average time per epoch using this line
                    # is 2 min 30 s while the current for-loop calculation is aound 2 min.
                    #sum =  logsumexp(backward[:,T-t] + feature_matrix[:,T-t])
                    sum = 0
                    for i in range(N):
                        if i == 0:
                            sum = backward[i,T-t] + feature_matrix[i, T-t]
                        else:
                            sum = np.logaddexp(sum, backward[i,T-t]+feature_matrix[i,T-t])
                    backward[s,T-t-1] = sum
            ### Termination step (skipped because we don't have the end state)

            # Z is in log space
            # NOTE: Originally we can choose aribitray column of states and then
            # use the formula on the slide "Computing Marginals" from Lec 5. 
            # However, we can choose the last column to avoid using backward
            # because backward are all zeros in the log space
            Z = 0
            for s in range(N):
                if s == 0:
                    Z = forward[s, -1]
                else:
                    Z = np.logaddexp(Z, forward[s,-1])
            # Z value should be the same for each word 
            # Z value is in log-space

            ### Compute P(y_i = s | X) (value in real-space)
            p_y_s_x = np.zeros(shape=(N,T))
            for s in range(N):
                for t in range(T):
                    p_y_s_x[s,t] = -np.exp(forward[s,t] + backward[s,t] - Z)

            ##  Compute the stochastic gradient of the feature vector for a sentence
            for word_idx in xrange(len(sentences[sentence_idx])):
                ### Find the gold tag for the given word
                gold_tag = tag_indexer.get_index(sentences[sentence_idx].get_bio_tags()[word_idx])
                features = feature_cache[sentence_idx][word_idx][gold_tag]
                loss += np.sum([feature_weights[i] for i in features])
                gradients.increment_all(features, 1)
                for tag_idx in xrange(N):
                    features = feature_cache[sentence_idx][word_idx][tag_idx]
                    gradients.increment_all(features, p_y_s_x[tag_idx, word_idx])

            ## Update the weights using the gradient computed
            loss -= Z
            for feature in gradients.keys():
                feature_weights[feature] += eta * gradients.get_count(feature)

        loss = -loss / num_sentences_use

        # Calculate the amount of time used for one epoch
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print('epoch: {} time eplased: {:0>2}:{:05.2f}'.format(str(epoch), int(minutes), seconds))
        
        epoch += 1
        if epoch % 10 == 0:
            eta = eta * 0.1
        
        if epoch >= numIters:
            break

        # Run bunch of experimentations to gather the data for plot
        if os.getenv('CRF_EXP', False):
            if os.getenv('CRF_ENG', False):
                # Gather the data on Accuracy vs. epoch on dev set
                crf_model = CrfNerModel(tag_indexer, feature_indexer, feature_weights)
                dev = read_data("data/eng.testa")
                dev_decoded = [crf_model.decode(test_ex) for test_ex in dev]
                print_evaluation(dev, dev_decoded)
            elif os.getenv('CRF_DEU', False):
                crf_model = CrfNerModel(tag_indexer, feature_indexer, feature_weights)
                deu_dev = read_data("data/deu.testa")
                deu_dev_decoded = [crf_model.decode(test_ex) for test_ex in deu_dev]
                print_evaluation(deu_dev, deu_dev_decoded)
    return CrfNerModel(tag_indexer, feature_indexer, feature_weights)


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
