#pragma once
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#include <pthread.h>
#include <fcntl.h>
#include <sys/stat.h>
using namespace std;

class Neuron {
public:
    Neuron(double data = 0.0) {
        this->data = data;
    }

    double data;
    vector<double> weights;
};

class Layer {
public:
    Layer() {
        pthread_mutex_init(&mutex, NULL);
    }

    void print() const {
        cout << endl;
        for (size_t i = 0; i < neurons.size(); i++) {
            cout << "Neuron " << (i + 1) << " data: " << neurons[i].data << endl;
            cout << "Weights: ";
            for (size_t j = 0; j < neurons[i].weights.size(); j++) {
                cout << neurons[i].weights[j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    int numNeurons() {
        return neurons.size();
    }

    vector<Neuron> neurons;
    pthread_mutex_t mutex;
};

class InputLayer : public Layer {
public:
    int count;
    InputLayer(int& numNeurons) {
        count = 0;
    	vector<vector<double>> inputArr = {
        {0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.1, -0.2},
        {-0.4, 0.5, 0.6, -0.4, 0.5, 0.6, -0.4, 0.5}};
        for (unsigned int i = 0; i < inputArr.size(); i++) {
            Neuron n;
            n.data = inputArr[i][0];
            n.weights = inputArr[i];
            neurons.push_back(n);
        }
    }
};

class HiddenLayer : public Layer {
public:
    HiddenLayer() {}
};

class OutputLayer : public Layer {
public:
    OutputLayer() {
        vector<double> outputArr = {-0.1, 0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8};
        for (double val : outputArr) {
            Neuron n;
            n.weights.push_back(val);
            neurons.push_back(std::move(n));
        }
    }
};


void createHiddenLayers(vector<HiddenLayer>& hiddenLayers);
void displayHiddenLayers(vector<HiddenLayer>& hiddenLayers);
void computeInnerLayer(InputLayer&, vector<double>&, vector<string>&, int);
void computeLayer(vector<HiddenLayer>&, OutputLayer&, vector<string>&, size_t);

struct ThreadArgs {
    Neuron neuron;
    vector<double>* layerValues;
    pthread_mutex_t* mutex;
    ThreadArgs() {}
};

void* computeLayerThread(void* args) {
    ThreadArgs* threadArgs = static_cast<ThreadArgs*>(args);
    Neuron& neuron = threadArgs->neuron;
    vector<double>& layerValues = *threadArgs->layerValues;
    pthread_mutex_t* mutex = threadArgs->mutex;

    for (size_t j = 0; j < neuron.weights.size(); j++) {
        pthread_mutex_lock(mutex);
        layerValues[j] += neuron.data * neuron.weights[j];
        pthread_mutex_unlock(mutex);
    }
    pthread_exit(NULL);
}

void computeInnerLayer(InputLayer& inputLayer, vector<double>& innerLayerValues, vector<string>& pipeNames, int inputLayerNeurons) {
    pthread_t* inputThreads = new pthread_t[inputLayerNeurons];
    ThreadArgs* threadArgsArray = new ThreadArgs[inputLayerNeurons];
    for (int i = 0; i < inputLayerNeurons; i++) {
        threadArgsArray[i].neuron = inputLayer.neurons[i];
        threadArgsArray[i].layerValues = &innerLayerValues;
        threadArgsArray[i].mutex = &inputLayer.mutex;
    }

    for (unsigned int i = 0; i < inputLayerNeurons; i++) {
        pthread_create(&inputThreads[i], NULL, computeLayerThread, &threadArgsArray[i]);
    }

    usleep(100000);

    int pipeFd = open(pipeNames[0].c_str(), O_WRONLY);
    write(pipeFd, innerLayerValues.data(), innerLayerValues.size() * sizeof(double));
    close(pipeFd);

    cout << endl << "processed by  inner layer" << endl;
    for (int i = 0; i < innerLayerValues.size(); i++) {
        cout << innerLayerValues[i] << "   ";
        innerLayerValues[i] = 0;
    }
    cout << endl << endl;

    delete[] inputThreads;
    delete[] threadArgsArray;
}

void computeLayer(vector<HiddenLayer>& hiddenLayers, OutputLayer& outputLayer, vector<string>& pipeNames, size_t numHiddenLayers) {
    for (size_t k = 0; k <= numHiddenLayers; k++) {
        pid_t pid = fork();

        if (pid > 0) {
            if (k == numHiddenLayers)
                break;
            continue;
        }

        if (k < numHiddenLayers) {
            vector<double> layerValues(hiddenLayers[k].neurons[k].weights.size());
            int pipeFdRead = open(pipeNames[k].c_str(), O_RDONLY);
            read(pipeFdRead, layerValues.data(), layerValues.size() * sizeof(double));
            close(pipeFdRead);

            for (size_t i = 0; i < layerValues.size(); i++) {
                hiddenLayers[k].neurons[i].data = layerValues[i];
                layerValues[i] = 0;
            }

            pthread_t* hiddenThreads = new pthread_t[layerValues.size()];
            ThreadArgs* threadArgsArray = new ThreadArgs[layerValues.size()];
            for (int i = 0; i < layerValues.size(); i++) {
                threadArgsArray[i].neuron = hiddenLayers[k].neurons[i];
                threadArgsArray[i].layerValues = &layerValues;
                threadArgsArray[i].mutex = &hiddenLayers[k].mutex;
            }

            for (unsigned int i = 0; i < layerValues.size(); i++) {
                pthread_create(&hiddenThreads[i], NULL, computeLayerThread, &threadArgsArray[i]);
            }

            usleep(100000);

            int pipeFdWrite = open(pipeNames[k + 1].c_str(), O_WRONLY);
            write(pipeFdWrite, layerValues.data(), layerValues.size() * sizeof(double));
            close(pipeFdWrite);

            cout << endl << "processed by Hidden Layer: " << (k + 1) << endl;
            for (int i = 0; i < layerValues.size(); i++) {
                cout << layerValues[i] << "   ";
                layerValues[i] = 0;
            }
            cout << endl << endl;

            layerValues.clear();
            layerValues.assign(2, 0.0);

            int backwardPipeFd = open(pipeNames[k + 1].c_str(), O_RDONLY);
            read(backwardPipeFd, layerValues.data(), layerValues.size() * sizeof(double));
            close(backwardPipeFd);

            cout << endl << "Backward propagation by Hidden Layer: " << (k + 1) << endl;
            for (int i = 0; i < layerValues.size(); i++) {
                cout << layerValues[i] << "   ";
            }
            cout << endl << endl;

            pipeFdWrite = open(pipeNames[k].c_str(), O_WRONLY);
            write(pipeFdWrite, layerValues.data(), layerValues.size() * sizeof(double));
            close(pipeFdWrite);

            delete[] hiddenThreads;
            delete[] threadArgsArray;
        } else {
            vector<double> layerValues(outputLayer.numNeurons());
            int pipeFdRead = open(pipeNames[k].c_str(), O_RDONLY);
            read(pipeFdRead, layerValues.data(), layerValues.size() * sizeof(double));
            close(pipeFdRead);

            for (size_t i = 0; i < layerValues.size(); i++) {
                outputLayer.neurons[i].data = layerValues[i];
            }

            for (size_t i = 0; i < layerValues.size(); i++) {
                layerValues[i] = 0;
            }

            pthread_t* outputThreads = new pthread_t[layerValues.size()];
            ThreadArgs* threadArgsArray = new ThreadArgs[layerValues.size()];
            for (int i = 0; i < layerValues.size(); i++) {
                threadArgsArray[i].neuron = outputLayer.neurons[i];
                threadArgsArray[i].layerValues = &layerValues;
                threadArgsArray[i].mutex = &outputLayer.mutex;
            }

            for (unsigned int i = 0; i < layerValues.size(); i++) {
                pthread_create(&outputThreads[i], NULL, computeLayerThread, &threadArgsArray[i]);
            }

            usleep(100000);

            cout << endl << "output Layer: " << endl;
            for (int i = 0; i < outputLayer.neurons[i].weights.size(); i++) {
                cout << layerValues[i] << "   ";
            }
            cout << endl << endl;

            double output = layerValues[0];
            layerValues.clear();
            layerValues.assign(2, 0.0);

            layerValues[0] = ((output * output) + output + 1) / 2.0;
            layerValues[1] = ((output * output) - output) / 2.0;

            int backwardPipeFd = open(pipeNames[k].c_str(), O_WRONLY);
            write(backwardPipeFd, layerValues.data(), layerValues.size() * sizeof(double));
            close(backwardPipeFd);

            delete[] outputThreads;
            delete[] threadArgsArray;
        }
        exit(0);
    }
}

void createHiddenLayers(vector<HiddenLayer>& hiddenLayers) {
    vector<vector<double>> hiddenArr1 = {
        {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
        {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
        {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
        {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
        {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
        {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
        {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
        {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8}
    };
    vector<vector<double>> hiddenArr2 = {
        {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
        {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
        {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
        {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
        {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
        {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
        {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
        {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8}
    };
    vector<vector<double>> hiddenArr3 = {
        {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
        {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},
        {0.6, -0.5, -0.7, 0.2, 0.4, 0.8, -0.1, -0.3},
        {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
        {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},
        {0.6, -0.5, -0.7, 0.2, 0.4, 0.8, -0.1, -0.3},
        {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
        {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1}
    };
    vector<vector<double>> hiddenArr4 = {
        {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
        {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2},
        {0.5, -0.4, -0.6, 0.3, 0.2, 0.8, -0.2, -0.1},
        {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
        {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2},
        {0.5, -0.4, -0.6, 0.3, 0.2, 0.8, -0.2, -0.1},
        {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
        {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2}
    };
    vector<vector<double>> hiddenArr5 = {
        {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
        {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1},
        {0.4, -0.3, -0.5, 0.1, 0.6, 0.7, -0.3, -0.2},
        {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
        {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1},
        {0.4, -0.3, -0.5, 0.1, 0.6, 0.7, -0.3, -0.2},
        {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
        {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1}
    };
    for (const auto& hiddenArr : {hiddenArr1, hiddenArr2, hiddenArr3, hiddenArr4, hiddenArr5}) {
        hiddenLayers.push_back(HiddenLayer());
        for (const auto& neuronWeights : hiddenArr) {
            Neuron n;
            for (double weight : neuronWeights) {
                n.weights.push_back(weight);
            }
            hiddenLayers.back().neurons.push_back(n);
        }
    }

    return;
}

void displayHiddenLayers(vector<HiddenLayer>& hiddenLayers) {
    for (auto& layer : hiddenLayers) {
        layer.Layer::print();
    }
}

int main(int argc, char* argv[]) {
    int inputLayerNeurons;
    do {
        system("clear");
        cout << "Enter the number of layers in your program: ";
        cin >> inputLayerNeurons;
    } while (inputLayerNeurons < 0);

    InputLayer inputLayer(inputLayerNeurons);
    vector<HiddenLayer> hiddenLayers;
    createHiddenLayers(hiddenLayers);
    size_t numHiddenLayers = hiddenLayers.size();

    OutputLayer outputLayer;

    vector<string> pipeNames(numHiddenLayers + 1);

    for (int i = 0; i <= numHiddenLayers; i++) {
        pipeNames[i] = "P" + to_string(i);
        const char* pipeName = pipeNames[i].c_str();
        int res = mkfifo(pipeName, 0666);
        if (res != 0) {
            cout << "Error :"<<endl<<" creating pipe caused failure" << pipeName << endl;
            return 1;
        }
    }

    for (int iteration = 0; iteration < 2; ++iteration) {
        cout << endl << " Pass Number : " << (iteration + 1) << endl;
        pid_t pid = fork();

        if (pid == 0) {
            computeLayer(hiddenLayers, outputLayer, pipeNames, numHiddenLayers);
            exit(0);
        } else if (pid > 0) {
            vector<double> innerLayerValues(inputLayer.neurons[0].weights.size(), 0);
            computeInnerLayer(inputLayer, innerLayerValues, pipeNames, inputLayerNeurons);

            innerLayerValues.clear();
            innerLayerValues.assign(2, 0.0);

            int backwardPipeFd = open(pipeNames[0].c_str(), O_RDONLY);
            read(backwardPipeFd, innerLayerValues.data(), innerLayerValues.size() * sizeof(double));
            close(backwardPipeFd);

            cout << endl << "Backward propagation by input layer" << endl;
            for (int i = 0; i < innerLayerValues.size(); i++) {
                cout << innerLayerValues[i] << "   ";
            }
            cout << endl << endl;

            inputLayer.neurons[0].data = innerLayerValues[0];
            inputLayer.neurons[1].data = innerLayerValues[1];
        } else {
            cerr << "fork failed" << endl;
            exit(EXIT_FAILURE);
        }
    }

    for (size_t i = 0; i <= numHiddenLayers; i++) {
        unlink(pipeNames[i].c_str());
    }

    pthread_exit(NULL);
}
