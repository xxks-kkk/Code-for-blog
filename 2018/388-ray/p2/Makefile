# Shortcuts for various tasks (UNIX only).
# To use a specific Python version run:
# $ make install PYTHON=python3.3

# python version
PYTHON=python
# input, embedded training directory
POS_TRAIN_I_E_DIR=pos_train_i_e_dir
# input, one_hot training directory
POS_TRAIN_I_O_DIR=pos_train_i_o_dir
# input, int training directory
POS_TRAIN_I_I_DIR=pos_train_i_i_dir
# output, embedded training directory
POS_TRAIN_O_E_DIR=pos_train_o_e_dir
# output, one_hot training directory
POS_TRAIN_O_O_DIR=pos_train_o_o_dir
# output, int training directory
POS_TRAIN_O_I_DIR=pos_train_o_i_dir
# baseline training directory
POS_TRAIN_B_DIR=pos_train_b_dir
# trace directory
TRACE_DIR=trace
# src directory location
SRC_DIR=src
# wsj data directory location
DATA_DIR=wsj

clean:
	rm -rf  $(POS_TRAIN_I_E_DIR) # input, embedded
	rm -rf  $(POS_TRAIN_I_O_DIR) # input, one_hot
	rm -rf  $(POS_TRAIN_I_I_DIR) # input, int
	rm -rf  $(POS_TRAIN_O_E_DIR) # output, embedded 
	rm -rf  $(POS_TRAIN_O_O_DIR) # output, one_hot
	rm -rf  $(POS_TRAIN_O_I_DIR) # output, int
	rm -rf  $(POS_TRAIN_B_DIR)   # baseline

setup:
	mkdir $(POS_TRAIN_I_E_DIR)
	mkdir $(POS_TRAIN_I_O_DIR)
	mkdir $(POS_TRAIN_I_I_DIR)
	mkdir $(POS_TRAIN_O_E_DIR)
	mkdir $(POS_TRAIN_O_O_DIR)
	mkdir $(POS_TRAIN_O_I_DIR)
	mkdir $(POS_TRAIN_B_DIR)
	mkdir -p $(TRACE_DIR)

# Including both training and testing
i_e:
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_E_DIR) standard train input embedded" > $(TRACE_DIR)/standard-input-embedded.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_E_DIR) standard train input embedded >> $(TRACE_DIR)/standard-input-embedded.txt
	printf "\n" >> $(TRACE_DIR)/standard-input-embedded.txt
	@echo "> $(PYTHON) $(HOME)/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir=$(POS_TRAIN_I_E_DIR)" >> $(TRACE_DIR)/standard-input-embedded.txt
	@echo "TensorBoard 1.6.0 at http://aida:6006 (Press CTRL+C to quit)" >> $(TRACE_DIR)/standard-input-embedded.txt
	printf "\n" >> $(TRACE_DIR)/standard-input-embedded.txt
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_E_DIR) standard test input embedded" >> $(TRACE_DIR)/standard-input-embedded.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_E_DIR) standard test input embedded >> $(TRACE_DIR)/standard-input-embedded.txt

i_o:
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_O_DIR)  standard train input one_hot" > $(TRACE_DIR)/standard-input-one_hot.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_O_DIR) standard train input one_hot >> $(TRACE_DIR)/standard-input-one_hot.txt
	printf "\n" >> $(TRACE_DIR)/standard-input-one_hot.txt
	@echo "> $(PYTHON) $(HOME)/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir=$(POS_TRAIN_I_O_DIR)" >> $(TRACE_DIR)/standard-input-one_hot.txt
	@echo "TensorBoard 1.6.0 at http://aida:6006 (Press CTRL+C to quit)" >> $(TRACE_DIR)/standard-input-one_hot.txt
	printf "\n" >> $(TRACE_DIR)/standard-input-one_hot.txt
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_O_DIR) standard test input one_hot" >> $(TRACE_DIR)/standard-input-one_hot.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_O_DIR) standard test input one_hot >> $(TRACE_DIR)/standard-input-one_hot.txt

i_i:
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_I_DIR) standard train input int" > $(TRACE_DIR)/standard-input-int.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_I_DIR) standard train input int >> $(TRACE_DIR)/standard-input-int.txt
	printf "\n" >> $(TRACE_DIR)/standard-input-int.txt
	@echo "> $(PYTHON) $(HOME)/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir=$(POS_TRAIN_I_I_DIR)" >> $(TRACE_DIR)/standard-input-int.txt
	@echo "TensorBoard 1.6.0 at http://aida:6006 (Press CTRL+C to quit)" >> $(TRACE_DIR)/standard-input-int.txt
	printf "\n" >> $(TRACE_DIR)/standard-input-int.txt
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_I_DIR) standard test input int" >> $(TRACE_DIR)/standard-input-int.txt
	$(PYTHON) pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_I_I_DIR) standard test input int >> $(TRACE_DIR)/standard-input-int.txt

o_e:
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_E_DIR) standard train output embedded" > $(TRACE_DIR)/standard-output-embedded.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_E_DIR) standard train output embedded >> $(TRACE_DIR)/standard-output-embedded.txt
	printf "\n" >> $(TRACE_DIR)/standard-output-embedded.txt
	@echo "> $(PYTHON) $(HOME)/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir=$(POS_TRAIN_O_E_DIR)" >> $(TRACE_DIR)/standard-output-embedded.txt
	@echo "TensorBoard 1.6.0 at http://aida:6006 (Press CTRL+C to quit)" >> $(TRACE_DIR)/standard-output-embedded.txt
	printf "\n" >> $(TRACE_DIR)/standard-output-embedded.txt
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_E_DIR) standard test output embedded" >> $(TRACE_DIR)/standard-output-embedded.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_E_DIR) standard test output embedded >> $(TRACE_DIR)/standard-output-embedded.txt

o_o:
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_O_DIR) standard train output one_hot" > $(TRACE_DIR)/standard-output-one_hot.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_O_DIR) standard train output one_hot >> $(TRACE_DIR)/standard-output-one_hot.txt
	printf "\n" >> $(TRACE_DIR)/standard-output-one_hot.txt
	@echo "> $(PYTHON) $(HOME)/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir=$(POS_TRAIN_O_O_DIR)" >> $(TRACE_DIR)/standard-output-one_hot.txt
	@echo "TensorBoard 1.6.0 at http://aida:6006 (Press CTRL+C to quit)" >> $(TRACE_DIR)/standard-output-one_hot.txt
	printf "\n" >> $(TRACE_DIR)/standard-output-one_hot.txt
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_O_DIR) standard test output one_hot" >> $(TRACE_DIR)/standard-output-one_hot.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_O_DIR) standard test output one_hot >> $(TRACE_DIR)/standard-output-one_hot.txt

o_i:
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_I_DIR) standard train output int" > $(TRACE_DIR)/standard-output-int.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_I_DIR) standard train output int >> $(TRACE_DIR)/standard-output-int.txt
	printf "\n" >> $(TRACE_DIR)/standard-output-int.txt
	@echo "> $(PYTHON) $(HOME)/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir=$(POS_TRAIN_O_I_DIR)" >> $(TRACE_DIR)/standard-output-int.txt
	@echo "TensorBoard 1.6.0 at http://aida:6006 (Press CTRL+C to quit)" >> $(TRACE_DIR)/standard-output-int.txt
	printf "\n" >> $(TRACE_DIR)/standard-output-int.txt
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_I_DIR) standard test output int" >> $(TRACE_DIR)/standard-output-int.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_O_I_DIR) standard test output int >> $(TRACE_DIR)/standard-output-int.txt

baseline:
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_B_DIR) standard train none" > $(TRACE_DIR)/standard-baseline.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_B_DIR) standard train none >> $(TRACE_DIR)/standard-baseline.txt
	printf "\n" >> $(TRACE_DIR)/standard-baseline.txt
	@echo "> $(PYTHON) $(HOME)/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir=$(POS_TRAIN_B_DIR)" >> $(TRACE_DIR)/standard-baseline.txt
	@echo "TensorBoard 1.6.0 at http://aida:6006 (Press CTRL+C to quit)" >> $(TRACE_DIR)/standard-baseline.txt
	printf "\n" >> $(TRACE_DIR)/standard-baseline.txt
	@echo "> $(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_B_DIR) standard test none" >> $(TRACE_DIR)/standard-baseline.txt
	$(PYTHON) $(SRC_DIR)/pos_bilstm.py $(DATA_DIR) $(POS_TRAIN_B_DIR) standard test none >> $(TRACE_DIR)/standard-baseline.txt	

all:
	@make i_e
	@make i_o
	@make i_i
	@make o_e
	@make o_o
	@make o_i
	@make baseline
