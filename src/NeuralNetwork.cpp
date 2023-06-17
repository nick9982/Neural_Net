#include "NeuralNetwork.hpp"

void NeuralNetwork::parse_input(int idx, string input)
{
    //Format: "Dense 512 act init"
    vector<string> split;
    int lastSpc = -1;
    for(int i = 0; i < input.length(); i++)
    {
        if(input[i] == 32)
        {
            split.push_back(input.substr(lastSpc+1, i-lastSpc-1));
            lastSpc = i;
        }
        else if(i == input.length()-1)
        {
            split.push_back(input.substr(lastSpc+1, i-lastSpc));
        }
    }

    for(int i = 0; i < split.size(); i++)
    {
        //cout << split[i] << endl;
    }

    int type = 1;
    if(idx == this->size-1)
    {
        type = 2;
        if(split[0] != "Dense")
            cerr << "Fatal error: output layer is not a dense layer!" << endl;
    }
    else if(idx == 0)
        type = 0;

    if(split[0] == "Dense")
    {
        int nodes = stoi(split[1]);
        lays[idx] = new DenseLayer(stoi(split[1]), StringActToInt(split[2]), StringInitToInt(split[3]), type, LastLayerNodes);
        LastLayerNodes = nodes;
        if(type == 0)
            this->input_size = nodes;
        else if(type == 2)
            this->output_size = nodes;
    }
    else if(split[0] == "Convolution")
    {
        
    }
    else if(split[0] == "Pooling")
    {
        
    }
    else if(split[0] == "Dropout")
    {
        
    }
}

int NeuralNetwork::StringActToInt(string input)
{
    if (input == "Linear")
        return 0;
    else if(input == "ReLU")
        return 1;
    else
        return 2;
}

int NeuralNetwork::StringInitToInt(string input)
{
    if(input == "Zero")
        return 0;
    else if(input == "Xavier")
        return 1;
    else
        return 2;
}

double* NeuralNetwork::forward(double* input)
{
    if(DenseLayer* dl = dynamic_cast<DenseLayer*>(this->lays[0]))
    {
        double* firstLayer = dl->getNeurons();
        for(int i = 0; i < this->input_size; i++)
            firstLayer[i] = input[i];
    }
    for(int i = 0; i < this->size-1; i++)
    {
        if(DenseLayer* NextDenseLayer = dynamic_cast<DenseLayer*>(this->lays[i+1]))
        {
            if(DenseLayer* CurrentDenseLayer = dynamic_cast<DenseLayer*>(this->lays[i]))
                CurrentDenseLayer->forward(NextDenseLayer->getNeurons());
        }
    }

    return dynamic_cast<DenseLayer*>(this->lays[this->size-1])->getNeurons(); //Make sure to ensure that the last layer is a dense layer
}

void NeuralNetwork::backward(double* errors)
{
    dynamic_cast<DenseLayer*>(this->lays[this->size-1])->firstDeltas(errors);

    for(int i = this->size-2; i > 0; i--)
    {
        if(DenseLayer* NextDenseLayer = dynamic_cast<DenseLayer*>(this->lays[i+1]))
        {
            if(DenseLayer* CurrentDenseLayer = dynamic_cast<DenseLayer*>(this->lays[i]))
                CurrentDenseLayer->backward(NextDenseLayer->getDeltas());
        }
    }
}

void NeuralNetwork::update()
{
    for(int i = 0; i < this->size-1; i++)
    {
        if(DenseLayer* NextDenseLayer = dynamic_cast<DenseLayer*>(this->lays[i+1]))
        {
            if(DenseLayer* CurrentDenseLayer = dynamic_cast<DenseLayer*>(this->lays[i]))
            {
                CurrentDenseLayer->update(NextDenseLayer->getDeltas());
            }
        }
    }
}
