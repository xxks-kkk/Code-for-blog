## How to run the code

We supply `trace.sh` that invokes all three models
and save the output to a directory called "trace".

You can also manually run the models:

- Run bigram model:

        # ATIS-3 corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BigramModel atis/ 0.1
        
        # Penn Treebank corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BigramModel wsj/ 0.1
        
        # Brown corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BigramModel brown/ 0.1
        
- Run backward bigram model:

        # ATIS-3 corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BackwardBigramModel atis/ 0.1
        
        # Penn Treebank corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BackwardBigramModel wsj/ 0.1
        
        # Brown corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BackwardBigramModel brown/ 0.1
        
- Run bidirectional bigram model:

        # ATIS-3 corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BidirectionalBigramModel atis/ 0.1
        
        # Penn Treebank corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BidirectionalBigramModel wsj/ 0.1
        
        # Brown corpus; use 0.1 of data as test set
        java -cp src/ nlp.lm.BidirectionalBigramModel brown/ 0.1
        
The above commands assume you have the following file structure:

```
.
├── p2.pdf
├── src
│   └── nlp
│       └── lm
│           ├── BackwardBigramModel.class
│           ├── BackwardBigramModel.java
│           ├── BidirectionalBigramModel.class
│           ├── BidirectionalBigramModel.java
│           ├── BigramModel.class
│           ├── BigramModel.java
│           ├── DoubleValue.class
│           ├── DoubleValue.java
│           ├── Ngram.class
│           ├── Ngram.java
│           ├── Ngrams.class
│           ├── POSTaggedFile.class
│           └── POSTaggedFile.java
├── trace
│   ├── backwardbigram-trace.txt
│   ├── bidirectionalbigram-trace.txt
│   └── bigram-trace.txt
|-- atis/
|-- wsj/
|-- brown/
└── trace.sh
```