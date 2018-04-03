#!/bin/sh

# This script is used to create
# 1. an initial training set
# 2. an "unlabeled" training pool for active learning
# 3. a test set

# The directory of WSJ Corpus
# We assume you're on UTCS machine
WSJ_DIR=/projects/nlp/penn-dependencybank/wsj-conllx
INIT_TRAIN_DIR=$WSJ_DIR/00
INIT_TRAIN_NAME=init_train.conllx
UNLABEL_TRAIN_NAME=unlabeled_train.conllx
TEST_NAME=test.conllx
# To create the initial training set, extract the first 50 sentences from section 00
INIT_TRAIN_SIZE=50


rm -rf $INIT_TRAIN_NAME; touch $INIT_TRAIN_NAME
rm -rf $UNLABEL_TRAIN_NAME; touch $UNLABEL_TRAIN_NAME
rm -rf $TEST_NAME; touch $TEST_NAME

# Create the initial training set
total_cnt=0
for filename in $INIT_TRAIN_DIR/*.conllx; do
    #printf "%s\n" $filename
    num_sent_in_file=`grep -cvP '\S' $filename`
    #printf "%d\n" $num_sent_in_file
    #printf "%d\n" $total_cnt
    #printf "%d\n" $(( $total_cnt + $num_sent_in_file ))
    if [ $(( $total_cnt + $num_sent_in_file )) -le $INIT_TRAIN_SIZE ]; then
        cat $filename >> $INIT_TRAIN_NAME
        total_cnt=$(($total_cnt + $num_sent_in_file))
    else
        break
    fi
done

# Create an "unlabeled" training pool for active learning
# For the unlabeled training set, concatenate sections 01-03 of WSJ
cat /projects/nlp/penn-dependencybank/wsj-conllx/01/* >> $UNLABEL_TRAIN_NAME
cat /projects/nlp/penn-dependencybank/wsj-conllx/02/* >> $UNLABEL_TRAIN_NAME
cat /projects/nlp/penn-dependencybank/wsj-conllx/03/* >> $UNLABEL_TRAIN_NAME

# For testing, use WSJ section 20
cat /projects/nlp/penn-dependencybank/wsj-conllx/20/* >> $TEST_NAME
