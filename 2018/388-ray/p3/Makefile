MAXITER=500
RESULT_DIR=results
MODELS_DIR=models

# Setup the necessary files
setup:
	./scripts/set.sh
	mkdir -p $(RESULT_DIR)
	mkdir -p $(MODELS_DIR)

# Run the random selection policy for active learning
random:
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile init_train.conllx \
		-testFile test.conllx \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/random_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain unlabeled_train.conllx \
		-policy random \
		-result $(RESULT_DIR)/result-$(MAXITER)-random.txt

# Run the sentence length selection policy for active learning
length:
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile init_train.conllx \
		-testFile test.conllx \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/length_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain unlabeled_train.conllx \
		-policy length \
		-result $(RESULT_DIR)/result-$(MAXITER)-length.txt

# Run the normalized Raw probability of the top parse selection policy for active learning
raw:
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile init_train.conllx \
		-testFile test.conllx \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/raw_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain unlabeled_train.conllx \
		-policy raw \
		-result $(RESULT_DIR)/result-$(MAXITER)-raw.txt

# Run the normalized Margin probability of the top parse selection policy for active learning
margin:
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile init_train.conllx \
		-testFile test.conllx \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/margin_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain unlabeled_train.conllx \
		-policy marign \
		-result $(RESULT_DIR)/result-$(MAXITER)-margin.txt

clean:
	rm -rf *.java
	rm -rf *.class
	ls | grep -e 'unlabeled_train.conllx[0-9]' | xargs rm
	ls | grep -e 'accum.conllx[0-9]' | xargs rm 
	ls | grep -e 'init_train.conllx\d' | xargs rm -rf


all:
	@make setup
	@make random
	@make setup
	@make length
	@make setup
	@make raw
	@make setup
	@make margin
