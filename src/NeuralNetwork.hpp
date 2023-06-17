#include <string>
#include <iostream>
#include <vector>
#include "DenseLayer.hpp"
#include "nnalgorithms.hpp"
using namespace std;

class NeuralNetwork
{
    public:
        template<typename... Args> NeuralNetwork(Args... args)
        {
            this->size = sizeof...(args)-1;
            lays = new Layer*[this->size];
            const char* params[this->size] = {args...};
            for(int i = this->size; i >= 0; i--)
            {
                parse_input(i, params[i]);   // Parsing the input parameter pack to initialize the layers
            }
        }
        double* forward(double*);
        void backward(double*);
        void update();
    private:
        Layer** lays;
        int size;
        int input_size;
        int output_size;
        int LastLayerNodes = 0;
        void parse_input(int, string);
        int StringActToInt(string);
        int StringInitToInt(string);
};
