CXXFLAGS = -std=c++11 -Wall -g

all: main

main: main.o 
	g++ $(CXXFLAGS) -o main main.o 

main.o: main.cpp 
	g++ $(CXXFLAGS) -c main.cpp

clean:
	rm -rf *.o main


