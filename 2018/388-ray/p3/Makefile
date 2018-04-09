MAXITER=500
RESULT_DIR=trace
MODELS_DIR=models
INIT_TRAIN_FILE=init_train.conllx
TEST_FILE=test.conllx
UNLABEL_TRAIN_FILE=unlabeled_train.conllx

RANDOM_LOG=$(RESULT_DIR)/result-$(MAXITER)-random.log
LENGTH_LOG=$(RESULT_DIR)/result-$(MAXITER)-length.log
RAW_LOG=$(RESULT_DIR)/result-$(MAXITER)-raw.log
MARGIN_LOG=$(RESULT_DIR)/result-$(MAXITER)-margin.log

default: random length raw margin

# Setup the necessary files
setup:
	cp $(INIT_TRAIN_FILE).bk $(INIT_TRAIN_FILE)
	cp $(TEST_FILE).bk $(TEST_FILE)
	cp $(UNLABEL_TRAIN_FILE).bk $(UNLABEL_TRAIN_FILE)
	mkdir -p $(RESULT_DIR)
	mkdir -p $(MODELS_DIR)

# Run the random selection policy for active learning
random:
	@make setup
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile $(INIT_TRAIN_FILE) \
		-testFile $(TEST_FILE) \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/random_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain $(UNLABEL_TRAIN_FILE) \
		-policy random \
		-result $(RESULT_DIR)/result-$(MAXITER)-random.txt

# Run the sentence length selection policy for active learning
length:
	@make setup
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile $(INIT_TRAIN_FILE) \
		-testFile $(TEST_FILE) \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/length_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain $(UNLABEL_TRAIN_FILE) \
		-policy length \
		-result $(RESULT_DIR)/result-$(MAXITER)-length.txt

# Run the normalized Raw probability of the top parse selection policy for active learning
raw:
	@make setup
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile $(INIT_TRAIN_FILE) \
		-testFile $(TEST_FILE) \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/raw_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain $(UNLABEL_TRAIN_FILE) \
		-policy raw \
		-result $(RESULT_DIR)/result-$(MAXITER)-raw.txt

# Run the normalized Margin probability of the top parse selection policy for active learning
margin:
	@make setup
	cp src/DependencyParserAPIUsage.java .
	javac -cp jars/*:src/*:. DependencyParserAPIUsage.java 
	java -cp jars/*:src/*:. DependencyParserAPIUsage \
		-trainFile $(INIT_TRAIN_FILE) \
		-testFile $(TEST_FILE) \
		-embedFile en-cw.txt \
		-model $(MODELS_DIR)/margin_model \
		-maxIter $(MAXITER) \
		-outFile annotations.conllx \
		-unlabelTrain $(UNLABEL_TRAIN_FILE) \
		-policy marign \
		-result $(RESULT_DIR)/result-$(MAXITER)-margin.txt

clean:
	rm -rf *.java
	rm -rf *.class
	ls | grep -e 'unlabeled_train.conllx[0-9]' | xargs rm
	ls | grep -e 'accum.conllx[0-9]' | xargs rm 
	ls | grep -e 'init_train.conllx\d' | xargs rm -rf

all:
	@echo "[start] random"
	@make setup
	@make random
	@echo "[start] length"
	@make setup
	@make length
	@echo "[start] raw"
	@make setup
	@make raw
	@echo "[start] margin"
	@make setup
	@make margin
