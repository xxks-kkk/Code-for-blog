# models.py

from utils import *
from adagrad_trainer import *
from treedata import *
import numpy as np

# Computes the sequence of decisions and ParserStates for a gold-standard sentence using the arc-standard
# transition framework. We use the minimum stack-depth heuristic, namely that
# Invariant: states[0] is the initial state. Applying decisions[i] to states[i] yields states[i+1].
def get_decision_sequence(parsed_sentence):
    decisions = []
    states = []
    state = initial_parser_state(len(parsed_sentence))
    while not state.is_finished():
        if not state.is_legal():
            raise Exception(repr(decisions) + " " + repr(state))
        # Look at whether left-arc or right-arc would add correct arcs
        if len(state.stack) < 2:
            result = "S"
        else:
            # Stack and buffer must both contain at least one thing
            one_back = state.stack_head()
            two_back = state.stack_two_back()
            # -1 is the ROOT symbol, so this forbids attaching the ROOT as a child of anything
            # (passing -1 as an index around causes crazy things to happen so we check explicitly)
            if two_back != -1 and parsed_sentence.get_parent_idx(two_back) == one_back and state.is_complete(two_back, parsed_sentence):
                result = "L"
            # The first condition should never be true, but doesn't hurt to check
            elif one_back != -1 and parsed_sentence.get_parent_idx(one_back) == two_back and state.is_complete(one_back, parsed_sentence):
                result = "R"
            elif len(state.buffer) > 0:
                result = "S"
            else:
                result = "R" # something went wrong, buffer is empty, just do right arcs to finish the tree
        decisions.append(result)
        states.append(state)
        if result == "L":
            state = state.left_arc()
        elif result == "R":
            state = state.right_arc()
        else:
            state = state.shift()
    states.append(state)
    return (decisions, states)

# Stores state of a shift-reduce parser, namely the stack, buffer, and the set of dependencies that have
# already been assigned. Supports various accessors as well as the ability to create new ParserStates
# from left_arc, right_arc, and shift.
class ParserState(object):
    # stack and buffer are lists of indices
    # The stack is a list with the top of the stack being the end
    # The buffer is a list with the first item being the front of the buffer (next word)
    # deps is a dictionary mapping *child* indices to *parent* indices
    # (this is the one-to-many map; parent-to-child doesn't work in map-like data structures
    # without having the values be lists)
    def __init__(self, stack, buffer, deps, children):
        self.stack = stack
        self.buffer = buffer
        self.deps = deps  # child -> parent
        self.children = children # parent -> list of children

    def __repr__(self):
        return repr(self.stack) + " " + repr(self.buffer) + " " + repr(self.deps)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.stack == other.stack and self.buffer == other.buffer and self.deps == other.deps
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def stack_len(self):
        return len(self.stack)

    def buffer_len(self):
        return len(self.buffer)

    def is_legal(self):
        return self.stack[0] == -1

    def is_finished(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

    def buffer_head(self):
        return self.get_buffer_word_idx(0)

    # Returns the buffer word at the given index
    def get_buffer_word_idx(self, index):
        if index >= len(self.buffer):
            raise Exception("Can't take the " + repr(index) + " word from the buffer of length " + repr(len(self.buffer)) + ": " + repr(self))
        return self.buffer[index]

    # Returns True if idx has all of its children attached already, False otherwise
    def is_complete(self, idx, parsed_sentence):
        _is_complete = True
        for child in range(0, len(parsed_sentence)):
            if parsed_sentence.get_parent_idx(child) == idx and (child not in self.deps.keys() or self.deps[child] != idx):
                _is_complete = False
        return _is_complete

    def stack_head(self):
        if len(self.stack) < 1:
            raise Exception("Can't go one back in the stack if there are no elements: " + repr(self))
        return self.stack[-1]

    def stack_two_back(self):
        if len(self.stack) < 2:
            raise Exception("Can't go two back in the stack if there aren't two elements: " + repr(self))
        return self.stack[-2]

    # Returns a new ParserState that is the result of taking the given action.
    # action is a string, either "L", "R", or "S"
    def take_action(self, action):
        if action == "L":
            return self.left_arc()
        elif action == "R":
            return self.right_arc()
        elif action == "S":
            return self.shift()
        else:
            raise Exception("No implementation for action " + action)

    # Returns a new ParserState that is the result of applying left arc to the current state. May crash if the
    # preconditions for left arc aren't met.
    def left_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_two_back(): self.stack_head()})
        new_children = dict(self.children)
        if self.stack_head() not in new_children.keys():
            new_children[self.stack_head()] = [self.stack_two_back()]
        else:
            new_children[self.stack_head()].append(self.stack_two_back())
        new_stack = list(self.stack[0:-2])
        new_stack.append(self.stack_head())
        return ParserState(new_stack, self.buffer, new_deps, new_children)

    # Returns a new ParserState that is the result of applying right arc to the current state. May crash if the
    # preconditions for right arc aren't met.
    def right_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_head(): self.stack_two_back()})
        new_children = dict(self.children)
        if self.stack_two_back() not in new_children.keys():
            new_children[self.stack_two_back()] = [self.stack_head()]
        else:
            new_children[self.stack_two_back()].append(self.stack_head())
        new_stack = list(self.stack[0:-1])
        return ParserState(new_stack, self.buffer, new_deps, new_children)

    # Returns a new ParserState that is the result of applying shift to the current state. May crash if the
    # preconditions for right arc aren't met.
    def shift(self):
        new_stack = list(self.stack)
        new_stack.append(self.buffer_head())
        return ParserState(new_stack, self.buffer[1:], self.deps, self.children)

    # Return the Dependency objects corresponding to the dependencies added so far to this ParserState
    def get_dep_objs(self, sent_len):
        dep_objs = []
        for i in range(0, sent_len):
            dep_objs.append(Dependency(self.deps[i], "?"))
        return dep_objs


# Returns an initial ParserState for a sentence of the given length. Note that because the stack and buffer
# are maintained as indices, knowing the words isn't necessary.
def initial_parser_state(sent_len):
    return ParserState([-1], range(0, sent_len), {}, {})


def get_label_indexer():
    label_indexer = Indexer()
    label_indexer.get_index("S")
    label_indexer.get_index("L")
    label_indexer.get_index("R")
    return label_indexer


def oracle(feature_indexer, feature_weights, state, sentence):
    # We perform logistic regression inference to figure out the decision for the current configuration
    num_decisions = 3
    label_indexer = get_label_indexer()

    feat_d_0 = extract_features(feature_indexer, sentence, state, label_indexer.get_object(0), False)
    weights_d_0 = np.take(feature_weights, feat_d_0)
    feat_d_1 = extract_features(feature_indexer, sentence, state, label_indexer.get_object(1), False)
    weights_d_1 = np.take(feature_weights, feat_d_1)
    feat_d_2 = extract_features(feature_indexer, sentence, state, label_indexer.get_object(2), False)        
    weights_d_2 = np.take(feature_weights, feat_d_2)

    # Calculate the $P(y='S'|x)$
    p_y_given_x_d_0 = np.exp(np.sum(weights_d_0)) / (np.exp(np.sum(weights_d_0)) + np.exp(np.sum(weights_d_1)) + np.exp(np.sum(weights_d_2)))
    # Calculate the $P(y='L'|x)$
    p_y_given_x_d_1 = np.exp(np.sum(weights_d_1)) / (np.exp(np.sum(weights_d_0)) + np.exp(np.sum(weights_d_1)) + np.exp(np.sum(weights_d_2)))
    # Calculate the $P(y='R'|x)$
    p_y_given_x_d_2 = np.exp(np.sum(weights_d_2)) / (np.exp(np.sum(weights_d_0)) + np.exp(np.sum(weights_d_1)) + np.exp(np.sum(weights_d_2)))

    cand = np.array([p_y_given_x_d_0, p_y_given_x_d_1, p_y_given_x_d_2])
    return np.argmax(cand)


class GreedyModel(object):
    def __init__(self, feature_indexer, feature_weights):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def parse(self, parsed_sentence):
        states = []
        state = initial_parser_state(len(parsed_sentence))
        while not state.is_finished():
            if not state.is_legal():
                raise Exception(repr(decisions) + " " + repr(state))
            if len(state.stack) < 2:
                result = "S"
            else:
                # Stack and buffer must both contain at least one thing
                one_back = state.stack_head()
                two_back = state.stack_two_back()
                # We consult Oracle to give us the right action to take: 'S','L' or 'R'
                result = oracle(self.feature_indexer, self.feature_weights, state, parsed_sentence)
                result = ["S", "L", "R"][result]
                # -1 is the ROOT symbol, so this forbids attaching the ROOT as a child of anything
                # (passing -1 as an index around causes crazy things to happen so we check explicitly)
                if two_back != -1 and result == "L":
                    result = "L"
                # The first condition should never be true, but doesn't hurt to check
                elif one_back != -1 and result == "R":
                    result = "R"
                elif result == "S" and len(state.buffer) > 0:
                    result = "S"
                else:
                    result = "R"  # something went wrong, buffer is empty, just do right arcs to finish the tree           
                states.append(state)
            if result == "L":
                state = state.left_arc()
            elif result == "R":
                state = state.right_arc()
            else:
                state = state.shift()
        states.append(state)
        return ParsedSentence(parsed_sentence.tokens, states[-1].get_dep_objs(len(parsed_sentence)))


def train_greedy_model(parsed_sentences):
    # We are training a classifier per state. In other words, we can batch up gradient updates on a per-sentence basis,
    # but the objective we are optimizing should be on the state level. 
    feature_indexer = Indexer()
    label_indexer = get_label_indexer()

    num_sentence = len(parsed_sentences)
    #num_sentence_use = 1000 
    num_sentence_use = num_sentence
    dev = read_data("data/dev.conllx")

    num_feature_per_extraction = 25
    # About the feature cache: what is the feature cache you're computing for the sentence?
    # Think about what features you actually need: if you're doing greedy, you should only be training on states that
    # are seen in the gold derivation, so only extract features for the (state, action) pairs that you'll actually consider.
    # We build feature cache first
    # a list of 3x23 matrics with each matrix represents features associated with three actions for a given state
    feature_cache = [] 
    decision_cache = []
    for sentence in parsed_sentences[0:num_sentence_use]:
        decisions, states = get_decision_sequence(sentence)
        num_states = len(states[:-1])
        for s in range(num_states):
            decision_cache.append(label_indexer.get_index(decisions[s]))        
            # 23 because 23 features added in "extract_features" function
            # 3  because there are three actions can take: 'S', 'L', 'R'
            feature_matrix = np.zeros(shape=(3,num_feature_per_extraction), dtype=int) 
            for d in range(len(label_indexer)):
                feat = extract_features(feature_indexer, sentence, states[s], label_indexer.get_object(d), True)
                feature_matrix[d,:] = feat
            feature_cache.append(feature_matrix)

    feature_cache_len = len(feature_cache)
    feature_weights = np.random.rand(len(feature_indexer))

    numEpoches = 30
    epoch = 0
    eta = 0.2 ## learning rate
    likelihood = 0
    while True:
        for i in range(feature_cache_len):
            feature_matrix = feature_cache[i]
            decision = decision_cache[i]
            # We use the logistic regression to build our Oracle            
            weights_matrix = np.zeros(shape=(3,num_feature_per_extraction))
            for d in range(len(label_indexer)):
                weights_matrix[d,:] = np.take(feature_weights, feature_matrix[d,:]) # This is the gold action: 'S', 'L', 'R'
            sum = 0
            for d in range(len(label_indexer)):
                sum += np.exp(np.sum(weights_matrix[d,:]))
            
            # We calculate the gradient here
            gradient_matrix = np.zeros(shape=(3,num_feature_per_extraction))
            for d in range(len(label_indexer)):
                if d == decision:
                    gradient_matrix[d,:] += 1  # This corresponds to $f_i(x_j, y_j^*)$
                p_y_given_x_d = np.exp(np.sum(weights_matrix[d,:])) / sum 
                gradient_matrix[d,:] -= p_y_given_x_d                

            # We maximize the log likelihood here
            for d in range(len(label_indexer)):
                feature_weights[feature_matrix[d,:]] += eta * gradient_matrix[d,0]

            # Calculate the likelihood to see whether it increases
            likelihood += (np.sum(feature_weights[feature_matrix[decision,:]]) - np.log(sum))

        print "likelihood: "
        print likelihood

        # Experiment
        trained_model = GreedyModel(feature_indexer, feature_weights)
        parsed_dev = [trained_model.parse(sent) for sent in dev]
        print_evaluation(dev, parsed_dev)

        epoch += 1
        if epoch % 10 == 0:
            eta = eta * 0.1
        if epoch >= numEpoches:
            break
    return GreedyModel(feature_indexer,feature_weights)

class BeamedModel(object):
    def __init__(self, feature_indexer, feature_weights, beam_size=3):
        self.label_indexer = get_label_indexer()
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.beam_size = beam_size

    def parse(self, sentence):
        def ssvm_score(feats, weights):
            score = 0.0
            for feat in feats:
                score += weights[feat]
            return score

        beam_list = []
        beam = Beam(self.beam_size)
        initial_state = initial_parser_state(len(sentence))
        feats = extract_features(self.feature_indexer, sentence, initial_state, "S", False)
        state = initial_state.shift()
        beam.add('S',ssvm_score(feats,self.feature_weights),state)
        beam_list.append(beam)

        num_possible_states = 2*len(sentence)
        while not len(beam_list[-1].get_actions()[0]) == num_possible_states:
            new_beam = Beam(self.beam_size)
            for actions, score in beam_list[-1].get_elts_and_scores():
                state = beam_list[-1].elts_to_states[actions]
                for d in range(len(self.label_indexer)):
                    action = self.label_indexer.get_object(d)
                    if action != "S" and len(state.stack) < 2:
                        continue
                    if action == "L" and len(state.stack) <= 2:
                        continue
                    if action == "S" and len(state.buffer) == 0:
                        continue
                    feats = extract_features(self.feature_indexer, sentence, state, action, False)
                    new_beam.add(actions+action, score+ssvm_score(feats,self.feature_weights), state.take_action(action))
            beam_list.append(new_beam)

        return ParsedSentence(sentence.tokens, beam_list[-1].elts_to_states[beam_list[-1].head()].get_dep_objs(len(sentence)))


def train_beamed_model(parsed_sentences):
    # Implement the structured SVM with the beam search
    beam_size = 3
    numEpoches = 10                           # iteration: per sentence; epoch: per all sentences
    epoch = 0
    batch_size = 20                           # we perform weights update every batch_size sentence
    lamb = 1.0E-5                             # parameter for adagrad
    eta = 5.0                                 # learning rate
    loss = 2.0                                # $l(y,y^*)$

    label_indexer = get_label_indexer()
    feature_indexer = Indexer()
    dev = read_data("data/dev.conllx")      # for the experiment purpose
    num_sentence = len(parsed_sentences)
    num_sentence_use = num_sentence
    #num_sentence_use = 10

    # We build this feature_cache to help us quickly visit all the features associated with gold actions and more importantly,
    # allow us to know the size of the feature_weigts we need to allocate beforehand. Thus, we only need to extract_features
    # using gold action and since there are 3 possible actions for each state, once we know the number of the gold states
    # we immediately know the size of feature_weights.
    feature_cache = [[[] for step in range(2*len(parsed_sentences[i]))] for i in range(len(parsed_sentences[:num_sentence_use]))]
    for sentence_idx, sentence in enumerate(parsed_sentences[:num_sentence_use]):
        decisions, states = get_decision_sequence(sentence)
        for action_idx in range(len(decisions)):
            feature_cache[sentence_idx][action_idx] = extract_features(feature_indexer, 
                                                                       sentence, 
                                                                       states[action_idx], 
                                                                       decisions[action_idx], 
                                                                       True)

    feature_weights = np.zeros(len(label_indexer)*len(feature_indexer),np.float)
    trainer = AdagradTrainer(feature_weights, lamb, eta)

    gradient = Counter()
    while True:
        print("# of epoch =", epoch)
        for sentence_idx, sentence in enumerate(parsed_sentences[:num_sentence_use]):
            if sentence_idx % batch_size == 0:
                trainer.apply_gradient_update(gradient, batch_size)
                # We want to resets the indices after each gradient update
                gradient = Counter()
            
            early_stopping = False
            gold_decisions, _ = get_decision_sequence(sentence)
            gold_decisions = ''.join(gold_decisions)

            beam_list = []
            # Initialization step 
            beam = Beam(beam_size)
            initial_state = initial_parser_state(len(parsed_sentences[sentence_idx]))
            feats = extract_features(feature_indexer, 
                                     sentence, 
                                     initial_state, 
                                     "S", 
                                     False)
            state = initial_state.shift()
            beam.add("S",trainer.score(feats),state)
            beam_list.append(beam)

            # Recursion step
            action_idx = 0
            while not len(beam_list[-1].get_actions()[0]) == len(gold_decisions):
                action_idx += 1
                new_beam = Beam(beam_size)
                for actions, score in beam_list[-1].get_elts_and_scores():
                    state = beam_list[-1].elts_to_states[actions]
                    for d in range(len(label_indexer)):
                        action = label_indexer.get_object(d)
                        if action != "S" and len(state.stack) < 2:
                            continue
                        if action == "L" and len(state.stack) <= 2:
                            continue
                        if action == "S" and len(state.buffer) == 0:
                            continue
                        feats = extract_features(feature_indexer, 
                                                 parsed_sentences[sentence_idx],
                                                 state, 
                                                 action, 
                                                 False)
                        new_beam.add(actions+action, score+trainer.score(feats), state.take_action(action))

                # The output of new_beam.get_actions() will be, for example, ['SSLSSLL', 'SSLSSSL', 'SSLSSLS']
                # If the gold decision sequence not in the predicted decision sequence list, then we perform early stop
                # action_idx + 1 because for example action_idx = 1, then [:(action_idx+1)] will be [0:2], which is the 0:1
                if gold_decisions[:(action_idx+1)] not in new_beam.get_actions():
                    for a in range(action_idx+1):
                        feats = feature_cache[sentence_idx][a]
                        gradient.add(Counter(feats))

                    # The output of new_beam.head() will be, for example, SSLSSLL
                    prediction = new_beam.head()
                    state = initial_parser_state(len(parsed_sentences[sentence_idx]))
                    for a in range(len(prediction)):
                        feats = extract_features(feature_indexer, 
                                                 parsed_sentences[sentence_idx],
                                                 state, 
                                                 prediction[a], 
                                                 False)
                        gradient.subtract(Counter(feats))
                        state = state.take_action(prediction[a])
                    early_stopping = True
                    break
                beam_list.append(new_beam)

            if not early_stopping:
                prediction = beam_list[-1].head()
                if gold_decisions == prediction:
                    # Here, we want to check the competing candidate of our prediction
                    # and make its weights lower than its originals so that our prediction
                    # weights become more significant
                    if len(beam_list[-1].get_actions()) > 1:
                        prediction = beam_list[-1].get_actions()[1]
                        if (beam_list[-1].scores[0] - beam_list[-1].scores[1]) > loss:
                            continue

                for a in range(len(gold_decisions)):
                    feats = feature_cache[sentence_idx][a]
                    gradient.add(Counter(feats))

                state = initial_parser_state(len(parsed_sentences[sentence_idx]))
                for a in range(len(prediction)):
                    feats = extract_features(feature_indexer, 
                                             parsed_sentences[sentence_idx],
                                             state, 
                                             prediction[a], 
                                             False)
                    gradient.subtract(Counter(feats))
                    state = state.take_action(prediction[a])

        # Experiments
        trained_model = BeamedModel(feature_indexer=feature_indexer, feature_weights=trainer.weights, beam_size=3)
        parsed_dev = [trained_model.parse(sent) for sent in dev]
        print_evaluation(dev, parsed_dev)

        epoch += 1
        if epoch >= numEpoches:
            break
    return BeamedModel(feature_indexer=feature_indexer, feature_weights=trainer.weights, beam_size=3)

# Extract features for the given decision in the given parser state. Features look at the top of the
# stack and the start of the buffer. Note that this isn't in any way a complete feature set -- play around with
# more of your own!
def extract_features(feat_indexer, sentence, parser_state, decision, add_to_indexer):
    feats = []
    sos_tok = Token("<s>", "<S>", "<S>")
    root_tok = Token("<root>", "<ROOT>", "<ROOT>")
    eos_tok = Token("</s>", "</S>", "</S>")
    two_back_idx = 0
    head_idx = 0
    
    if parser_state.stack_len() >= 1:
        head_idx = parser_state.stack_head()
        stack_head_tok = sentence.tokens[head_idx] if head_idx != -1 else root_tok
        if parser_state.stack_len() >= 2:
            two_back_idx = parser_state.stack_two_back()
            stack_two_back_tok = sentence.tokens[two_back_idx] if two_back_idx != -1 else root_tok
        else:
            stack_two_back_tok = sos_tok
    else:
        stack_head_tok = sos_tok
        stack_two_back_tok = sos_tok
    buffer_first_tok = sentence.tokens[parser_state.get_buffer_word_idx(0)] if parser_state.buffer_len() >= 1 else eos_tok
    buffer_second_tok = sentence.tokens[parser_state.get_buffer_word_idx(1)] if parser_state.buffer_len() >= 2 else eos_tok
    # Shortcut for adding features
    def add_feat(feat):
        maybe_add_feature(feats, feat_indexer, add_to_indexer, feat)
    add_feat(decision + ":S0Word=" + stack_head_tok.word)
    add_feat(decision + ":S0Pos=" + stack_head_tok.pos)
    add_feat(decision + ":S0CPos=" + stack_head_tok.cpos)
    add_feat(decision + ":S1Word=" + stack_two_back_tok.word)
    add_feat(decision + ":S1Pos=" + stack_two_back_tok.pos)
    add_feat(decision + ":S1CPos=" + stack_two_back_tok.cpos)
    add_feat(decision + ":B0Word=" + buffer_first_tok.word)
    add_feat(decision + ":B0Pos=" + buffer_first_tok.pos)
    add_feat(decision + ":B0CPos=" + buffer_first_tok.cpos)
    add_feat(decision + ":B1Word=" + buffer_second_tok.word)
    add_feat(decision + ":B1Pos=" + buffer_second_tok.pos)
    add_feat(decision + ":B1CPos=" + buffer_second_tok.cpos)
    add_feat(decision + ":S1S0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos)
    add_feat(decision + ":S0B0Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1B0Pos=" + stack_two_back_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B1Pos=" + stack_head_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":B0B1Pos=" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":S0B0WordPos=" + stack_head_tok.word + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0PosWord=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1S0WordPos=" + stack_two_back_tok.word + "&" + stack_head_tok.pos)
    add_feat(decision + ":S1S0PosWord=" + stack_two_back_tok.pos + "&" + stack_head_tok.word)
    add_feat(decision + ":S1S0B0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0B1Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)


    # additional feature
    add_feat(decision + ":S1S0WordDis=" + stack_two_back_tok.word + "&" + stack_head_tok.word + "&" + str(head_idx - two_back_idx))
    add_feat(decision + ":S1S0PosDis=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos + "&" + str(head_idx - two_back_idx))

    return feats

