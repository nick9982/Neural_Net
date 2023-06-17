#include "DenseLayer.hpp"
#include "nnalgorithms.hpp"

DenseLayer::DenseLayer(int nodes, int activation, int initialization, int type, int output) : Layer()
{
    this->input = nodes;
    this->output = output;
    this->type = type;

    this->neuron = new double[nodes];
    this->cache = new double[nodes];
    this->delta = new double[nodes];
    this->weight = new double[output*nodes];
    this->bias = 0;

    switch(activation)
    {
        case 0:
            this->act_function = Linear;
            this->act_function_derivative = LinearDerivative;
            break;
        case 1:
            this->act_function = ReLU;
            this->act_function_derivative = ReLUDerivative;
            break;
    }

    for(int i = 0; i < output*nodes; i++)
    {
        this->weight[i] = HeRandomInNormal(nodes);
    }
}

double* DenseLayer::getNeurons()
{
    return this->neuron;
}

double* DenseLayer::getDeltas()
{
    return this->delta;
}

void DenseLayer::forward(double* NextLayerNeurons)
{
    int wStart = 0;
    for(int i = 0; i < this->input; i++)
        this->neuron[i] = this->act_function(this->neuron[i]);

    for(int i = 0; i < this->output; i++)
    {
        double sum = 0;
        for(int j = 0; j < this->input; j++)
        {
             sum += this->neuron[j] * this->weight[wStart+i];
        }
        NextLayerNeurons[i] = sum + this->bias;
        wStart += this->input;
    }
}

void DenseLayer::firstDeltas(double *errors)
{
    for(int i = 0; i < this->input; i++)
        this->delta[i] = (this->neuron[i] - errors[i]) * this->act_function_derivative(this->cache[i]);
}

void DenseLayer::backward(double* NextLayerDeltas)
{
    int wStart = 0;
    for(int i = 0; i < this->input; i++)
    {
        double sum = 0;
        double neuron_derivative = this->act_function_derivative(cache[i]);
        for(int j = 0; j < this->output; j++)
            sum += weight[wStart + i] * delta[j] * neuron_derivative;
        delta[i] = sum;
        wStart += this->output;
    }
}

void DenseLayer::update(double* NextLayerDeltas)
{
    int wStart = 0;
    double deltaSum = 0;
    for(int i = 0; i < this->input; i++)
    {
        for(int j = 0; j < this->output; j++)
        {
            deltaSum += NextLayerDeltas[j];
            double gradient = this->neuron[i] * NextLayerDeltas[j];
            this->weight[wStart+j] = gradient * 0.001; //Learning rate is 0.001 temporarily. for now only using SGD
        }
        wStart += this->output;
    }
    this->bias -= deltaSum * 0.001;
}
