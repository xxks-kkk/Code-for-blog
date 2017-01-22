C_COMPILER=gcc
CFLAGS = -Wall -g -fstack-usage

SRC_FILES=\
  $(wildcard ./*.c) \

INC_DIRS=\

LIBS=\

OUTPUT_FILE=main

.PHONY: clean
.PHONY: test

test:
	$(C_COMPILER) $(CFLAGS) $(INC_DIRS) $(LIBS) $(SRC_FILES) -o $(OUTPUT_FILE)


clean:
	rm -rf $(OUTPUT_FILE)
