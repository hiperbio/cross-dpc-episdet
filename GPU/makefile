CC = dpcpp
FLAGS = -g -O2

all: v1 v2 v3 v4

v1: v1.cpp
	$(CC) -o $@ $^ $(FLAGS)
v2: v2.cpp
	$(CC) -o $@ $^ $(FLAGS)
v3: v3.cpp
	$(CC) -o $@ $^ $(FLAGS)
v4: v4.cpp
        $(CC) -o $@ $^ $(FLAGS)

clean:
	rm v1 v2 v3 v4