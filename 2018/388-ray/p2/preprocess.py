import glob

MAX_LENGTH = 100

# http://teacher.scholastic.com/reading/bestpractices/vocabulary/pdf/prefixes_suffixes.pdf
COMMON_SUFFIX = {
    "able":1,
    "ible":2,
    "al":3,
    "ial":4,
    "ed":5,
    "en":6,
    "er":7,
    "er,":8,
    "est":9,
    "ful":10,
    "ic":11,
    "ing":12,
    "ion":13,
    "tion":14,
    "ation":15,
    "ition":16,
    "ity":17,
    "ty":18,
    "ive":19,
    "ative":20,
    "itive":21,
    "less":22,
    "ly":23,
    "ment":24,
    "ness":25,
    "ous":26,
    "eous":27,
    "ious":28,
    "s":29,
    "es":30,
    "y":31
}

COMMON_PREFIX = {
    "anti":32,
    "de":33,
    "dis":34,
    "en":35,
    "em":36,
    "fore":37,
    "in":38,
    "im":39,
    "il":40,
    "ir":41,
    "inter":42,
    "mid":43,
    "mis":44,
    "non":45,
    "over":46,
    "pre":47,
    "re":48,
    "semi":49,
    "sub":50,
    "super":51,
    "trans":52,
    "un":53,
    "under":54
}

class PreprocessData:
    def __init__(self, dataset_type='wsj'):
        self.vocabulary = {}
        self.pos_tags = {}
        self.prefix_orthographic = {}
        self.suffix_orthographic = {}
        self.dataset_type = dataset_type

        self.prefix_orthographic['none'] = 0
        self.suffix_orthographic['none'] = 0

    def isCapitalized(self, word, mode="strict"):
        """
        Check whether a given word is capitalized
        :param word:
        :param mode: - loose: any character in a word has upper case counts
                     - strict: Only beginning char of a word in upper case counts
        :return: 1 - True; 0 - False
        """
        if mode == "loose":
            if any(x.isupper() for x in word):
                return 1
        elif mode == "strict":
            return 1 if word[0].isupper() else 0
        return 0

    def containsHypen(self, word):
        """
        Check whether a given word contains hyphen
        :param word:
        :return: 1 - True; 0 - False
        """
        return 1 if "-" in word else 0

    def startsWithNumber(self, word):
        """
        Check whether a given word starts with a number
        :param word:
        :return: 1 - True; 0 - False
        """
        return 1 if word[0].isdigit() else 0

    def get_prefix_feature_id(self, token, mode):
        for prefix in COMMON_PREFIX:
            if token.startswith(prefix):
                return self.get_orthographic_id(prefix, self.prefix_orthographic)
        return self.prefix_orthographic['none']

    def get_suffix_feature_id(self, token, mode):
        for suffix in COMMON_SUFFIX:
            if token.endswith(suffix):
                return self.get_orthographic_id(suffix, self.suffix_orthographic)
        return self.prefix_orthographic['none']

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

    ## OOV words represented by len(vocab)
    def get_oov_id(self, dic):
        return len(dic)

    def get_pad_id(self, dic):
        return len(self.vocabulary) + 1

    ## We add the feature to the map and assign a new id no matter whether we are in train or test
    def get_orthographic_id(self, pos, dic):
        if pos not in dic:
            dic[pos] = len(dic)
        return dic[pos]

    ## get id of given token(pos) from dictionary dic.
    ## if not in dic, extend the dic if in train mode
    ## else use representation for unknown token
    def get_id(self, pos, dic, mode):
        if pos not in dic:
            if mode == 'train':
                dic[pos] = len(dic)
            else:
                return self.get_oov_id(dic)
        return dic[pos]

    ## Process single file to get raw data matrix
    def processSingleFile(self, inFileName, mode):
        matrix = []
        row = []
        word_count = 0
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
                            word_count = 0
                            row = []
                            break
                        elif PreprocessData.isFeasibleStartingCharacter(token[0]):
                            wordPosPair = token.split('/')
                            # The MAX_LENGTH check ensures that the training vocabulary
                            # only includes in those words that are finally a part of the
                            # training instance (not the clipped off portion)
                            if len(wordPosPair) == 2 and word_count < MAX_LENGTH:
                                word_count += 1
                                word, tag = wordPosPair[0], wordPosPair[1]
                                ## get ids for word
                                feature = self.get_id(word, self.vocabulary, mode)

                                ## get ids for prefix and suffix features
                                prefix_feature = self.get_prefix_feature_id(word, mode)
                                suffix_feature = self.get_suffix_feature_id(word, mode)

                                ## get ids for capitalized feature
                                cap_feature = self.isCapitalized(word)

                                ## get ids for whether the word starts with a number feature
                                num_feature = self.startsWithNumber(word)

                                ## get ids for whether the word contains a hyphen feature
                                hyphen_feature = self.containsHypen(word)

                                # get id for pos tag. Instead of passing input mode
                                # we pass train as the mode so that we can include all pos tags
                                row.append((feature,
                                            self.get_id(tag, self.pos_tags, 'train'),
                                            prefix_feature,
                                            suffix_feature,
                                            cap_feature,
                                            num_feature,
                                            hyphen_feature))
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
        XP = []
        XS = []
        XC = []
        XN = []
        XH = []
        original_len = len(mat)
        mat = filter(lambda x: len(x) <= max_size, mat)
        no_removed = original_len - len(mat)
        for row in mat:
            X_row = [tup[0] for tup in row]
            y_row = [tup[1] for tup in row]
            XP_row = [tup[2] for tup in row]
            XS_row = [tup[3] for tup in row]
            XC_row = [tup[4] for tup in row]
            XN_row = [tup[5] for tup in row]
            XH_row = [tup[5] for tup in row]
            ## padded words represented by len(vocab) + 1
            X_row = X_row + [self.get_pad_id(self.vocabulary)] * (max_size - len(X_row))
            ## Padded pos tags represented by -1
            y_row = y_row + [-1] * (max_size - len(y_row))
            XP_row = XP_row + [self.prefix_orthographic['none']] * (max_size - len(XP_row))
            XS_row = XS_row + [self.suffix_orthographic['none']] * (max_size - len(XS_row))
            XC_row = XC_row + [self.suffix_orthographic['none']] * (max_size - len(XC_row))
            XN_row = XN_row + [self.prefix_orthographic['none']] * (max_size - len(XN_row))
            XH_row = XH_row + [self.prefix_orthographic['none']] * (max_size - len(XH_row))
            X.append(X_row)
            y.append(y_row)
            XP.append(XP_row)
            XS.append(XS_row)
            XC.append(XC_row)
            XN.append(XN_row)
            XH.append(XH_row)
        return X, y, XP, XS, XC, XN, XH, no_removed
