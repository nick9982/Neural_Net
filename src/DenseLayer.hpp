#include "Layer.hpp"
#include <iostream>
using namespace std;

class DenseLayer : public virtual Layer
{
    public:
        DenseLayer(int, int, int, int, int);
        void forward(double*);
        void firstDeltas(double*);
        void backward(double*);
        void update(double*);
        double* getNeurons();
        double* getDeltas();
    private:
        int output, input, type;
        double *neuron, *cache, *delta, *weight, bias;
};
