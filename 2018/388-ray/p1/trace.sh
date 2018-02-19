#!/bin/sh -vx

# check if "trace" directory exists. If not, create one
mkdir -p trace

exe () {
    params="$@"                       # Put all of the command-line into "params"
    printf "$ $params" >> "$SCRIPT_LOG"   # Print the command to the log file
    $params                           # Execute the command
    }

javac -cp src/ src/nlp/lm/BigramModel.java

SCRIPT_LOG="trace/bigram-trace.txt"

# echo "$ java -cp src/ nlp.lm.BigramModel atis/ 0.1" > trace/bigram-trace.txt
# java -cp src/ nlp.lm.BigramModel atis/ 0.1 >> trace/bigram-trace.txt
# echo "" >> trace/bigram-trace.txt
exe java -cp src/ nlp.lm.BigramModel atis/ 0.1


echo "$ java -cp src/ nlp.lm.BigramModel wsj/ 0.1" >> trace/bigram-trace.txt
java -cp src/ nlp.lm.BigramModel wsj/ 0.1 >> trace/bigram-trace.txt
echo "" >> trace/bigram-trace.txt

echo "$ java -cp src/ nlp.lm.BigramModel brown/ 0.1" >> trace/bigram-trace.txt
java -cp src/ nlp.lm.BigramModel brown/ 0.1 >> trace/bigram-trace.txt
echo "" >> trace/bigram-trace.txt

javac -cp src/ src/nlp/lm/BackwardBigramModel.java

echo "$ java -cp src/ nlp.lm.BackwardBigramModel atis/ 0.1" > trace/backwardbigram-trace.txt
java -cp src/ nlp.lm.BackwardBigramModel atis/ 0.1 >> trace/backwardbigram-trace.txt
echo "" >> trace/bigram-trace.txt

echo "$ java -cp src/ nlp.lm.BackwardBigramModel wsj/ 0.1" >> trace/backwardbigram-trace.txt
java -cp src/ nlp.lm.BackwardBigramModel wsj/ 0.1 >> trace/backwardbigram-trace.txt
echo "" >> trace/bigram-trace.txt

echo "$ java -cp src/ nlp.lm.BackwardBigramModel brown/ 0.1" >> trace/backwardbigram-trace.txt
java -cp src/ nlp.lm.BackwardBigramModel brown/ 0.1 >> trace/backwardbigram-trace.txt
echo "" >> trace/bigram-trace.txt

javac -cp src/ src/nlp/lm/BidirectionalBigramModel.java

echo "$ java -cp src/ nlp.lm.BidirectionalBigramModel atis/ 0.1" > trace/bidirectionalbigram-trace.txt
java -cp src/ nlp.lm.BidirectionalBigramModel atis/ 0.1 >> trace/bidirectionalbigram-trace.txt
echo "" >> trace/bigram-trace.txt

echo "$ java -cp src/ nlp.lm.BidirectionalBigramModel wsj/ 0.1" > trace/bidirectionalbigram-trace.txt
java -cp src/ nlp.lm.BidirectionalBigramModel wsj/ 0.1 >> trace/bidirectionalbigram-trace.txt
echo "" >> trace/bigram-trace.txt

echo "$ java -cp src/ nlp.lm.BidirectionalBigramModel brown/ 0.1" > trace/bidirectionalbigram-trace.txt
java -cp src/ nlp.lm.BidirectionalBigramModel brown/ 0.1 >> trace/bidirectionalbigram-trace.txt
echo "" >> trace/bigram-trace.txt
