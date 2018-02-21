#!/bin/sh -vx

# check if "trace" directory exists. If not, create one
mkdir -p trace

javac -cp src/ src/nlp/lm/BigramModel.java

echo "$ java -cp src/ nlp.lm.BigramModel atis/ 0.1" > trace/bigram-trace.txt
java -cp src/ nlp.lm.BigramModel atis/ 0.1 >> trace/bigram-trace.txt
printf "\n" >> trace/bigram-trace.txt

echo "$ java -cp src/ nlp.lm.BigramModel wsj/ 0.1" >> trace/bigram-trace.txt
java -cp src/ nlp.lm.BigramModel wsj/ 0.1 >> trace/bigram-trace.txt
printf "\n" >> trace/bigram-trace.txt

echo "$ java -cp src/ nlp.lm.BigramModel brown/ 0.1" >> trace/bigram-trace.txt
java -cp src/ nlp.lm.BigramModel brown/ 0.1 >> trace/bigram-trace.txt
printf "\n" >> trace/bigram-trace.txt

javac -cp src/ src/nlp/lm/BackwardBigramModel.java

echo "$ java -cp src/ nlp.lm.BackwardBigramModel atis/ 0.1" > trace/backwardbigram-trace.txt
java -cp src/ nlp.lm.BackwardBigramModel atis/ 0.1 >> trace/backwardbigram-trace.txt
printf "\n" >> trace/backwardbigram-trace.txt

echo "$ java -cp src/ nlp.lm.BackwardBigramModel wsj/ 0.1" >> trace/backwardbigram-trace.txt
java -cp src/ nlp.lm.BackwardBigramModel wsj/ 0.1 >> trace/backwardbigram-trace.txt
printf "\n" >> trace/backwardbigram-trace.txt

echo "$ java -cp src/ nlp.lm.BackwardBigramModel brown/ 0.1" >> trace/backwardbigram-trace.txt
java -cp src/ nlp.lm.BackwardBigramModel brown/ 0.1 >> trace/backwardbigram-trace.txt
printf "\n" >> trace/backwardbigram-trace.txt

javac -cp src/ src/nlp/lm/BidirectionalBigramModel.java

echo "$ java -cp src/ nlp.lm.BidirectionalBigramModel atis/ 0.1" > trace/bidirectionalbigram-trace.txt
java -cp src/ nlp.lm.BidirectionalBigramModel atis/ 0.1 >> trace/bidirectionalbigram-trace.txt
printf "\n" >> trace/bidirectionalbigram-trace.txt

echo "$ java -cp src/ nlp.lm.BidirectionalBigramModel wsj/ 0.1" >> trace/bidirectionalbigram-trace.txt
java -cp src/ nlp.lm.BidirectionalBigramModel wsj/ 0.1 >> trace/bidirectionalbigram-trace.txt
printf "\n" >> trace/bidirectionalbigram-trace.txt

echo "$ java -cp src/ nlp.lm.BidirectionalBigramModel brown/ 0.1" >> trace/bidirectionalbigram-trace.txt
java -cp src/ nlp.lm.BidirectionalBigramModel brown/ 0.1 >> trace/bidirectionalbigram-trace.txt
printf "\n" >> trace/bidirectionalbigram-trace.txt

exit 0
