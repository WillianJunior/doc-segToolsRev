CC=g++
DEBUG=-g -ggdb -gdwarf
OPENCV=`pkg-config --cflags --libs opencv`

all: cc.o main.cpp
	$(CC) $(DEBUG) main.cpp cc.o -o segment.out $(OPENCV)

cc.o: ConnComponents.cpp ConnComponents.hpp
	$(CC) $(DEBUG) -c ConnComponents.cpp -o cc.o