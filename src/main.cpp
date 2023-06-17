#include "NeuralNetwork.hpp"
#include "DataMining/power_consumption.hpp"
#include <chrono>

void learnPowerConsumption();

int main(int argc, char *argv[])
{
    /* NeuralNetwork nn("Dense 5 ReLU Xavier", "Dense 10 ReLU Xavier", "Dense 5 ReLU Xavier"); */
    /*  */
    /* double *input = new double[5]{0, 1, 2, 3, 4}; */
    /* double *output = nn.forward(input); */
    /* input[2] = input[2]-1; */
    /* nn.backward(input); */
    /* nn.update(); */
    /*  */
    /* for(int i = 0; i < 5; i++) */
    /* { */
    /*     cout << "out: " << output[i] << endl; */
    /* } */
    learnPowerConsumption();
}

void learnPowerConsumption()
{
    dataset processedData(32000, "../src/DataMining/data/tetuanCityPowerConsumption.csv", "Tetuan City Power Consumption");
    cout << "processing data" << endl;
    processedData.shuffle();

    vector<dataset> train_test_data = processedData.split(26000, "train_data", "test_data");
    dataset train_data = train_test_data[0];
    dataset test_data = train_test_data[1];

    NeuralNetwork nn(
        "Dense 6 Linear HeRandom",
        "Dense 35 ReLU HeRandom",
        "Dense 35 ReLU HeRandom",
        "Dense 35 ReLU HeRandom",
        "Dense 35 ReLU HeRandom",
        "Dense 3 Linear HeRandom"
    );
    double *input = new double[6];
    double *output = new double[3];

    double avg = 0;
    int avg_cnt = 0;
    double total = 0;
    int view_cnt = 1;
    cout << "testing initial performance..." << endl;



    for(uint i = 0; i < 200; i++)
    {
        for(uint j = 0; j < test_data.data[0].size(); j++)
        {
            if(j < 6) input[j] = test_data.data[i][j];
            else output[j-6] = test_data.data[i][j];
        }
    
        double* nno = nn.forward(input);
    
        double sum = 0;
        for(uint i = 0; i < 3; i++)
        {
            sum += abs(test_data.minMaxUnnormalization(nno[i], i+5) - test_data.minMaxUnnormalization(output[i], i+5));
        }
        cout << "error: " << sum/3 << endl;
        avg_cnt++;
        total += sum/3;
        avg = total/avg_cnt;
    }

    cout << "training..." << endl;
    auto start = chrono::_V2::high_resolution_clock::now();
    for(uint i = 0; i < train_data.data.size(); i++)
    {
        for(uint j = 0; j < train_data.data[i].size(); j++)
        {
            if(j < 6) input[j] = train_data.data[i][j];
            else output[j-6] = train_data.data[i][j];
        }
    
        nn.forward(input);
        nn.backward(output);
        nn.update();
    
        if(i % 1000 == 0) cout << "[" << i/1000 << "/"<<ceil(train_data.data.size()/1000)<<"]" << endl;
    }
    cout << "[" << ceil(train_data.data.size()/1000) << "/"<<ceil(train_data.data.size()/1000)<<"]" << endl;
    auto stop = chrono::_V2::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
    
    double error_before_training = avg;
    avg = 0;
    avg_cnt = 0;
    total = 0;
    view_cnt = 1;
    cout << "\ntesting final performance..." << endl;
    for(uint i = 0; i < test_data.data.size(); i++)
    {
        for(uint j = 0; j < test_data.data[0].size(); j++)
        {
            if(j < 6) input[j] = test_data.data[i][j];
            else output[j-6] = test_data.data[i][j];
        }
    
        double* nno = nn.forward(input);
    
        double sum = 0.f;
        for(uint i = 0; i < 3; i++)
        {
            sum += abs(test_data.minMaxUnnormalization(nno[i], i+5) - test_data.minMaxUnnormalization(output[i], i+5));
        }
        avg_cnt++;
        total += sum/3;
        avg = total/avg_cnt;
    }
    cout << "\nAverage error before training: " << error_before_training << endl;
    cout << "Average error after training: " << avg << endl;
    cout << "\nThe network's predictions are " << (1 - (avg/error_before_training)) * 100 << " percent more accurate than randomly choosing. " << endl;
    cout << "\nThe error is the average difference between the network's prediction of\nthe three region's power consumption and the actual power consumption." << endl;
    if(duration.count() * 0.000001 >= 60) cout << "\nTraining time: " << (duration.count() * 0.000001)/60 << " minutes" << endl;
    else cout << "\nTraining time: " << duration.count() * 0.000001 << " seconds" << endl;
}
