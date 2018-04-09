## How to run the program

You have several options to run the program. Please don't touch the shipped `test.conllx.bk`, 
`unlabeled_train.conllx.bk`, and `init_train.conllx.bk`. Those are test file, initial training files,
and unlabeled training sentences pool. Those files can only be recreated via `set.sh` if you have access
to UTCS lab machines. 

### Run all the experiments

1. Type `script` to log all the terminal output (this will help to collect the full trace
of training from stdout) and run `make -j4`. This probably the fastest way to run the 
experiments.

2. Type `script` again and run `make all`. The advantage of this option is that all
experiments are run sequentially and you may get a clean full trace in `typescript` file.

### Run each individual experiment

1. Type `script` and run:

    - random: `make random`
    - raw: `make raw`
    - length: `make length`
    - margin: `make margin`

2. Run the Java program directly:

```
usage: java -cp stanford-corenlp-jar/*:src/*:. DependencyParserAPIUsage.java
 -embedFile <arg>       Path to embedding file
 -h,--help              Print out help manual
 -maxIter <arg>         maxIter property for Stanford Neural Network
                        Dependency Parser
 -model <arg>           Path where model is to be saved
 -numSentInInit <arg>   Number of sentences you want to use from initial
                        training set
 -numSentInPool <arg>   Number of sentences you want to pick from
                        "unlabeled" training pool in each iteration
 -outFile <arg>         Path where test data annotations are stored
 -output <arg>          Name of the file that you want to save the stdout
                        to
 -policy <arg>          Selection policy in the active learning
 -result <arg>          Name of the file that the result is to be saved
 -testFile <arg>        Path to test file
 -trainFile <arg>       Path to training file
 -unlabelTrain <arg>    Path to unlabeled training instances
```

For example, if I want to run the program with initial training set file path `init_train.conllx`,
"unlabeled" training pool `unlabeled_train.conllx`, test set path `test.conllx`, number of sentences
to use in the inital training set `2`, and number of sentences to use in the "unlabeled" training set
`1`, I would run:
    
    
    cp src/DependencyParserAPIUsage.java . ; \
    javac -cp jars/*:src/*:. DependencyParserAPIUsage.java; \
    java -cp jars/*:src/*:. DependencyParserAPIUsage \
                            -trainFile init_train.conllx \
                            -testFile test.conllx \
                            -embedFile en-cw.txt \
                            -model results/random_model \
                            -maxIter 500 \
                            -outFile annotations.conllx \
                            -unlabelTrain unlabeled_train.conllx \
                            -policy random \
                            -numSentInPool 1 \
                            -numSentInInit 2 \
                            -result results/result-500-random.txt &> results/result-random-500.log
    

## Note

1. Before running `Makefile`, make sure you want to backup any initial training file, "unlabeled" training
pool, and test file with the file names with extra `.bk` extension in the same directory as your oginal files. 
`make setup` will assume the existence and the location of those backup files.

2. `Makefile` current doesn't support `-numSentInPool` and `-numSentInInit` options. Run the Java program
directly if you really want to use these two options.

3. There are no safe guard against `-numSentInPool` and `-numSentInInit`. In other words, make sure you
specify the number of sentences within the maximum sentences number of the files.

4. To reproduce the experiments in the writeup, use the `Makefile` with default values.

5. "results-batch2" is for the second run of experiments mentioned in the writeup. Results shown in
Table 1 is from the ".txt" and ".log" files immediately under "trace" directory.

## Directory Structure

```
.
├── emnlp2016.pdf
├── en-cw.txt
├── init_train.conllx
├── init_train.conllx.bk
├── jars
│   ├── commons-cli-1.4.jar
│   └── stanford-corenlp.jar
├── Makefile
├── README.md
├── scripts
│   ├── plot.py
│   ├── set.sh
│   └── split.awk
├── src
│   ├── DependencyParserAPIUsage.class
│   └── DependencyParserAPIUsage.java
├── test.conllx
├── test.conllx.bk
├── trace
│   ├── learning-curves
│   │   ├── curve.png
│   │   ├── curves-no-dash.png
│   │   ├── length.png
│   │   ├── margin.png
│   │   ├── random.png
│   │   └── raw.png
│   ├── result-500-length.log
│   ├── result-500-length.txt
│   ├── result-500-margin.log
│   ├── result-500-margin.txt
│   ├── result-500-random.log
│   ├── result-500-random.txt
│   ├── result-500-raw.log
│   ├── result-500-raw.txt
│   ├── results-batch2
│   │   ├── result-500-length.txt
│   │   ├── result-500-margin.txt
│   │   ├── result-500-random.txt
│   │   └── result-500-raw.txt
│   └── typescript
├── unlabeled_train.conllx
└── unlabeled_train.conllx.bk
```

