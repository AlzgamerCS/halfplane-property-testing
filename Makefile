# Define the compiler
CXX = g++

# Define compiler flags
CXXFLAGS = -Wall -std=c++23 -O3

# Define the target executable
TARGET = main

# Define the source files
SRCS = generator.cpp utility.cpp

# Define the header files (not strictly necessary, but good practice)
HEADERS = utility.h

# Default target: build the executable
$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean target: remove the executable
clean:
	rm -f $(TARGET)