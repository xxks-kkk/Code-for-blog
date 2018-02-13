# Makefile for the CS:APP Shell Lab

TEAM = NOBODY
VERSION = 1
DRIVER = ./sdriver.pl
MSH = ./msh
MSHREF = ./mshref
MSHARGS = "-p"
CC = gcc
CFLAGS = -Wall -O2
FILES = $(MSH) ./myspin ./mysplit ./mystop ./myint ./fib ./handle ./mykill ./psh

all: $(FILES)

msh: msh.o util.o jobs.o
	$(CC) $(CFLAGS) msh.o util.o jobs.o -o msh


psh: psh.o util.o 
	$(CC) $(CFLAGS) psh.o util.o -o psh

handle: handle.o util.o
	$(CC) $(CFLAGS) handle.o util.o -o handle

mykill: mykill.o util.o
	$(CC) $(CFLAGS) mykill.o util.o -o mykill


##############################
# Prepare your work for upload
##############################
FILENAME = shell_project.tar

turnin.tar: clean
	tar cvf $(FILENAME) `find . -type f | grep -v \.tar$$ | grep -v \.tar\.gz$$ | grep -v /\.git | grep -v \.swp$$ | grep -v CVS | grep -v svn | grep -v \.pl$$ | grep -v mshref | grep -v ~$$`
	gzip $(FILENAME)

turnin: turnin.tar
	@echo "Created $(FILENAME).gz for submission.  Please upload to Canvas."
	@echo "Before uploading, please verify:"
	@echo "		Your README is correctly filled out."
	@echo "		Your pair programming log is in the project directory."
	@echo "If either of those items are not done, please update your submission and run the make turnin command again."
	@ls -al $(FILENAME).gz


##################
# Regression tests
##################

# Run tests using the student's shell program
test01:
	$(DRIVER) -t trace01.txt -s $(MSH) -a $(MSHARGS)
test02:
	$(DRIVER) -t trace02.txt -s $(MSH) -a $(MSHARGS)
test03:
	$(DRIVER) -t trace03.txt -s $(MSH) -a $(MSHARGS)
test04:
	$(DRIVER) -t trace04.txt -s $(MSH) -a $(MSHARGS)
test05:
	$(DRIVER) -t trace05.txt -s $(MSH) -a $(MSHARGS)
test06:
	$(DRIVER) -t trace06.txt -s $(MSH) -a $(MSHARGS)
test07:
	$(DRIVER) -t trace07.txt -s $(MSH) -a $(MSHARGS)
test08:
	$(DRIVER) -t trace08.txt -s $(MSH) -a $(MSHARGS)
test09:
	$(DRIVER) -t trace09.txt -s $(MSH) -a $(MSHARGS)
test10:
	$(DRIVER) -t trace10.txt -s $(MSH) -a $(MSHARGS)
test11:
	$(DRIVER) -t trace11.txt -s $(MSH) -a $(MSHARGS)
test12:
	$(DRIVER) -t trace12.txt -s $(MSH) -a $(MSHARGS)
test13:
	$(DRIVER) -t trace13.txt -s $(MSH) -a $(MSHARGS)
test14:
	$(DRIVER) -t trace14.txt -s $(MSH) -a $(MSHARGS)
test15:
	$(DRIVER) -t trace15.txt -s $(MSH) -a $(MSHARGS)
test16:
	$(DRIVER) -t trace16.txt -s $(MSH) -a $(MSHARGS)

# Run the tests using the reference shell program
rtest01:
	$(DRIVER) -t trace01.txt -s $(MSHREF) -a $(MSHARGS)
rtest02:
	$(DRIVER) -t trace02.txt -s $(MSHREF) -a $(MSHARGS)
rtest03:
	$(DRIVER) -t trace03.txt -s $(MSHREF) -a $(MSHARGS)
rtest04:
	$(DRIVER) -t trace04.txt -s $(MSHREF) -a $(MSHARGS)
rtest05:
	$(DRIVER) -t trace05.txt -s $(MSHREF) -a $(MSHARGS)
rtest06:
	$(DRIVER) -t trace06.txt -s $(MSHREF) -a $(MSHARGS)
rtest07:
	$(DRIVER) -t trace07.txt -s $(MSHREF) -a $(MSHARGS)
rtest08:
	$(DRIVER) -t trace08.txt -s $(MSHREF) -a $(MSHARGS)
rtest09:
	$(DRIVER) -t trace09.txt -s $(MSHREF) -a $(MSHARGS)
rtest10:
	$(DRIVER) -t trace10.txt -s $(MSHREF) -a $(MSHARGS)
rtest11:
	$(DRIVER) -t trace11.txt -s $(MSHREF) -a $(MSHARGS)
rtest12:
	$(DRIVER) -t trace12.txt -s $(MSHREF) -a $(MSHARGS)
rtest13:
	$(DRIVER) -t trace13.txt -s $(MSHREF) -a $(MSHARGS)
rtest14:
	$(DRIVER) -t trace14.txt -s $(MSHREF) -a $(MSHARGS)
rtest15:
	$(DRIVER) -t trace15.txt -s $(MSHREF) -a $(MSHARGS)
rtest16:
	$(DRIVER) -t trace16.txt -s $(MSHREF) -a $(MSHARGS)


# clean up
clean:
	rm -f $(FILES) *.o *~ *.bak *.BAK



