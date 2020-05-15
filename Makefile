CC = gcc
LD = gcc
CFLAGS = -Wall -O0 -fopenmp
LDFLAGS = #-lpthread -lm -fopenmp
SRCS := $(wildcard *.c) # wildcard
OBJS = $(SRCS:.c=.o)
DEPS = $(SRCS:.c=.dep)
EXEC = $(SRCS:.c=)
RM = rm -f

all: $(EXEC)

adaboost: adaboost.o
	$(LD) $(LDFLAGS) -fopenmp -lm -o $@ $^

adaboost.o: adaboost.h calloc_errchk.h adaboost_io.h bit_op.h

adaboost.h: calloc_errchk.h bit_op.h
adaboost_io.h: calloc_errchk.h


clean:
	$(RM) $(OBJS) $(EXEC) *~

.PHONY:
	all clean
