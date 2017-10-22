# treedata.py


# Abstraction to bundle words with POS and chunks for featurization
# pos = part-of-speech, cpos = coarse part-of-speech
class Token:
    def __init__(self, word, pos, cpos):
        self.word = word
        self.pos = pos
        self.cpos = cpos

    def __repr__(self):
        return self.word


# Bundles the two components of a syntactic dependency: the index (0-based) of the
# parent and the label. A parent index of -1 indicates attaching to the root.
# For unlabeled dependencies, you can set the label to any placeholder value or
# the empty string.
class Dependency:
    def __init__(self, parent_idx, label):
        self.parent_idx = parent_idx
        self.label = label

    def __repr__(self):
        return repr(self.parent_idx) + "(" + self.label + ")"


# Wrapper over a sentence which bundles a list of tokens with a parallel list of Dependency objects
# representing the parse.
class ParsedSentence:
    def __init__(self, tokens, deps=None):
        self.tokens = tokens
        self.deps = deps

    def __repr__(self):
        return repr([repr(tok) for tok in self.tokens]) + "\n" + repr([repr(dep) for dep in self.deps])

    def __len__(self):
        return len(self.tokens)

    # Returns a list of Dependency objects corresponding to the parse
    def get_deps(self):
        return self.deps

    # Returns the parent index of the word at position idx
    def get_parent_idx(self, idx):
        if idx < 0:
            raise Exception("idx must be >=0 but was " + repr(idx) + "; -1 is the ROOT symbol, which has no parent")
        return self.deps[idx].parent_idx

    # Returns the parent label of the word at position idx
    def get_parent_label(self, idx):
        if idx < 0:
            raise Exception("idx must be >=0 but was " + repr(idx) + "; -1 is the ROOT symbol, which has no parent")
        return self.deps[idx].label


# Reads a treebank in the CoNLL-X format from the given file. Returns a list of ParsedSentences
def read_data(file):
    f = open(file)
    sentences = []
    curr_tokens = []
    curr_deps = []
    for line in f:
        stripped = line.strip()
        if stripped != "":
            fields = stripped.split()
            if len(fields) == 10:
                curr_tokens.append(Token(fields[1], fields[3], fields[4]))
                if fields[6] != "_":
                    curr_deps.append(Dependency(int(fields[6]) - 1, fields[7]))
        elif stripped == "" and len(curr_tokens) > 0:
            sentences.append(ParsedSentence(curr_tokens, curr_deps))
            curr_tokens = []
            curr_deps = []
    return sentences


# Prints the unlabeled and labeled attachment scores (UAS and LAS) with the predictions from guess_parsed_sentences
# evaluated based on the gold annotations in gold_parsed_sentences. These are both lists of ParsedSentences. Note that
# UAS does not depend on labels -- your parser does not need to produce labels.
def print_evaluation(gold_parsed_sentences, guess_parsed_sentences):
    correct_unlabeled = 0
    correct_labeled = 0
    total = 0
    for gold, guess in zip(gold_parsed_sentences, guess_parsed_sentences):
        for index in xrange(0, len(gold)):
            if len(gold.get_deps()) != len(guess.get_deps()):
                print "FUCK"
            if gold.get_deps()[index] == guess.get_deps()[index]:
                correct_labeled += 1
                correct_unlabeled += 1
            elif gold.get_parent_idx(index) == guess.get_parent_idx(index):
                correct_unlabeled += 1
            total += 1
    print "UAS: " + repr(correct_unlabeled) + "/" + repr(total) + " = " + "{0:.2f}".format((correct_unlabeled/float(total)) * 100) + \
          "; LAS: " + repr(correct_labeled) + "/" + repr(total) + " = " + "{0:.2f}".format((correct_labeled / float(total)) * 100)


# Writes parsed_sentences (a list of ParsedSentences) to outfile in the CoNLL format
def print_output(parsed_sentences, outfile):
    f = open(outfile, 'w')
    for sentence in parsed_sentences:
        for i in xrange(0, len(sentence)):
            tok = sentence.tokens[i]
            f.write(repr(i+1) + "\t" + tok.word + "\t_\t" + tok.cpos + "\t" + tok.pos + "\t_\t" +
                    repr(sentence.get_parent_idx(i)+1) + "\t" + sentence.get_parent_label(i) + "\t_\t_\n")
        f.write("\n")
    f.close()
