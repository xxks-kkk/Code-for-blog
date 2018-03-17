import glob
from random import shuffle

MAX_LENGTH = 100

# http://teacher.scholastic.com/reading/bestpractices/vocabulary/pdf/prefixes_suffixes.pdf
COMMON_SUFFIX = (
    "able",
    "ible",
    "al",
    "ial",
    "ed",
    "en",
    "er",
    "er,",
    "est",
    "ful",
    "ic",
    "ing",
    "ion",
    "tion",
    "ation",
    "ition",
    "ity",
    "ty",
    "ive",
    "ative",
    "itive",
    "less",
    "ly",
    "ment",
    "ness",
    "ous",
    "eous",
    "ious",
    "s",
    "es",
    "y"
)

COMMON_PREFIX = (
    "anti",
    "de",
    "dis",
    "en",
    "em",
    "fore",
    "in",
    "im",
    "il",
    "ir",
    "inter",
    "mid",
    "mis",
    "non",
    "over",
    "pre",
    "re",
    "semi",
    "sub",
    "super",
    "trans",
    "un",
    "under"
)


class PreprocessData:
    """
    Preprocess the WSJ data to generate the input and output vectors that are fed into the BiLSTM
    """
    def __init__(self, dataset_type='wsj'):
        self.vocabulary = {}
        self.pos_tags = {}
        self.dataset_type = dataset_type

    def isCapitalized(self, word, mode="loose"):
        """
        Check whether a given word is capitalized
        :param word:
        :param mode: - loose: any character in a word has upper case counts
                     - strict: Only beginning char of a word in upper case counts
        :return: bool
        """
        if mode == "loose":
            if any(x.isupper() for x in word):
                return True
        elif mode == "strict":
            return word[0].isupper()
        return False

    def containsSuffix(self, word):
        """
        Check whether a given word contains the predefined suffix
        :param word:
        :return: bool
        """
        return word.endswith(COMMON_SUFFIX)

    def containsPrefix(self, word):
        """
        Check whether a given word contains the predefined prefix
        :param word:
        :return: bool
        """
        return word.startswith(COMMON_PREFIX)

    ## Get standard split for WSJ
    def get_standard_split(self, files):
        if self.dataset_type == 'wsj':
            train_files = []
            val_files = []
            test_files = []
            for file_ in files:
                partition = int(file_.split('/')[-2])
                if partition >= 0 and partition <= 18:
                    train_files.append(file_)
                elif partition <= 21:
                    val_files.append(file_)
                else:
                    test_files.append(file_)
            return train_files, val_files, test_files
        else:
            raise Exception('Standard Split not Implemented for ' + self.dataset_type)

    @staticmethod
    def isFeasibleStartingCharacter(c):
        unfeasibleChars = '[]@\n'
        return not (c in unfeasibleChars)

    ## unknown words represented by len(vocab)
    def get_unk_id(self, dic):
        return len(dic)

    def get_pad_id(self, dic):
        return len(self.vocabulary) + 1

    ## get id of given token(pos) from dictionary dic.
    ## if not in dic, extend the dic if in train mode
    ## else use representation for unknown token
    def get_id(self, pos, dic, mode):
        if pos not in dic:
            if mode == 'train':
                dic[pos] = len(dic)
            else:
                return self.get_unk_id(dic)
        return dic[pos]

    ## Process single file to get raw data matrix
    def processSingleFile(self, inFileName, mode):
        matrix = []
        row = []
        with open(inFileName) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    pass
                else:
                    tokens = line.split()
                    for token in tokens:
                        ## ==== indicates start of new example
                        if token[0] == '=':
                            if row:
                                matrix.append(row)
                            row = []
                            break
                        elif PreprocessData.isFeasibleStartingCharacter(token[0]):
                            wordPosPair = token.split('/')
                            if len(wordPosPair) == 2:
                                word, tag = wordPosPair[0], wordPosPair[1]
                                ## get ids for word and pos tag
                                feature = self.get_id(word, self.vocabulary, mode)
                                # include all pos tags.
                                # Add the feature: (word, isCapitalized, containsPrefix, containsSuffix, pos)
                                row.append((feature,
                                            self.isCapitalized(word),
                                            self.containsPrefix(word),
                                            self.containsSuffix(word),
                                            self.get_id(tag, self.pos_tags, 'train')))
        if row:
            matrix.append(row)
        return matrix

    ## get all data files in given subdirectories of given directory
    def preProcessDirectory(self, inDirectoryName, subDirNames=['*']):
        if not (subDirNames):
            files = glob.glob(inDirectoryName + '/*.pos')
        else:
            files = [glob.glob(inDirectoryName + '/' + subDirName + '/*.pos')
                     for subDirName in subDirNames]
            files = set().union(*files)
        return list(files)

    ## Get basic data matrix with (possibly) variable sized senteces, without padding
    def get_raw_data(self, files, mode):
        matrix = []
        for f in files:
            matrix.extend(self.processSingleFile(f, mode))
        return matrix

    def split_data(self, data, fraction):
        split_index = int(fraction * len(data))
        left_split = data[:split_index]
        right_split = data[split_index:]
        if not (left_split):
            raise Exception('Fraction too small')
        if not (right_split):
            raise Exception('Fraction too big')
        return left_split, right_split

    ## Get rid of sentences greater than max_size
    ## and pad the remaining if less than max_size
    def get_processed_data(self, mat, max_size):
        X = []
        y = []
        original_len = len(mat)
        mat = filter(lambda x: len(x) <= max_size, mat)
        no_removed = original_len - len(mat)
        for row in mat:
            X_row = [tup[0] for tup in row]
            y_row = [tup[1] for tup in row]
            ## padded words represented by len(vocab) + 1
            ## Padding is used so the feature vector for each sentence is of same length.
            ## Since sentences are not all of equal length we need padding at the end to achieve this.
            X_row = X_row + [self.get_pad_id(self.vocabulary)] * (max_size - len(X_row))
            ## Padded pos tags represented by -1
            y_row = y_row + [-1] * (max_size - len(y_row))
            X.append(X_row)
            y.append(y_row)
        return X, y, no_removed
