***Multi-Layer Perceptron (MLP) Implementation***

**Overview:**
This C++ code implements a Multi-Layer Perceptron (MLP) neural network. It consists of input, hidden, and output layers, with the ability to specify the number of hidden layers during runtime. The program utilizes multi-threading for concurrent computation within layers and inter-process communication (IPC) using named pipes for communication between layers.

**Dependencies:**
* C++11 compiler
* POSIX compliant operating system (Linux/Unix)
* Compile the code using any C++11 compliant compiler:  (g++ -o NN NN.cpp -lpthread)
* Run the compiled executable: (./NN)

**Features:**
- Input Layer: Takes input values and distributes them to the neurons in the layer.
- Hidden Layer: Processes input data using multi-threading and forwards results to subsequent layers.
- Output Layer: Generates output based on processed data from the preceding layers.
- Concurrency: Utilizes pthreads for parallel computation within layers.
- Inter-Process Communication (IPC): Communicates between layers using named pipes for data transfer.
- Backward Propagation: Implements backward propagation for updating weights during training.
**Classes:**
- Neuron: Represents a single neuron in the network with a data value and associated weights.
- Layer: Abstract class representing a layer in the network, with mutex for thread safety.
- InputLayer: Subclass of Layer, initializes input data and distributes it to neurons.
- HiddenLayer: Subclass of Layer, processes data using multi-threading.
- OutputLayer: Subclass of Layer, generates output based on processed data.
**Functions:**
- createHiddenLayers: Creates and initializes hidden layers with specified weights.
- displayHiddenLayers: Displays the neurons and weights in hidden layers.
- computeInnerLayer: Computes the inner layer's output values using multi-threading.
- computeLayer: Computes output for hidden and output layers, implementing forward and backward propagation.
- computeLayerThread: Thread function for computing neuron output values.
- main: Entry point of the program, orchestrates the execution of the MLP.
