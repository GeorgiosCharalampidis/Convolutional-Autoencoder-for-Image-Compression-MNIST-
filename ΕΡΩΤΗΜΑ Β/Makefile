# Compiler settings
CXX = g++
CXXFLAGS = -Wall -g -std=c++17 -w  # Added C++17 standard and suppress all warnings

# Executable name
TARGET = graph_search

# Object files
OBJS = mnist.o lsh_class.o Hypercube.o graph.o global_functions.o graph_search.o MRNGGraph.o

# Header files
HEADERS = Hypercube.h lsh_class.h graph.h mnist.h global_functions.h MRNGGraph.h

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Individual file dependencies
mnist.o: mnist.cpp mnist.h
	$(CXX) $(CXXFLAGS) -c mnist.cpp

lsh_class.o: lsh_class.cpp lsh_class.h global_functions.h
	$(CXX) $(CXXFLAGS) -c lsh_class.cpp

Hypercube.o: Hypercube.cpp Hypercube.h
	$(CXX) $(CXXFLAGS) -c Hypercube.cpp

graph.o: graph.cpp graph.h lsh_class.h Hypercube.h mnist.h global_functions.h
	$(CXX) $(CXXFLAGS) -c graph.cpp

global_functions.o: global_functions.cpp global_functions.h
	$(CXX) $(CXXFLAGS) -c global_functions.cpp

graph_search.o: graph_search.cpp graph.h lsh_class.h Hypercube.h mnist.h global_functions.h MRNGGraph.h
	$(CXX) $(CXXFLAGS) -c graph_search.cpp

# Updated rule for MRNGGraph
MRNGGraph.o: MRNGGraph.cpp MRNGGraph.h global_functions.h
	$(CXX) $(CXXFLAGS) -c MRNGGraph.cpp

# Clean rule
clean:
	rm -f $(TARGET) $(OBJS)
