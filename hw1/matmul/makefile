all: matmul

matmul: matmul.cpp
	g++ -Xpreprocessor -fopenmp -lomp matmul.cpp -o matmul

clean:
	rm matmul