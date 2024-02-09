#ifndef NETWORK_H__
#define NETWORK_H__

#include "data.hh"
#include "neuron.hh"
#include "layer.hh"
#include "iris_handler.hh"
#include "common.hh"

class Network : public common_data
{
  public:
  std::vector<Layer *> Layers;
  double               learningRate;
  double               testPerfomance;

  Network(std::vector<int> hiddenLayerSpec, int inputSize, int numClasses, double learningRate);
  ~Network();

  std::vector<double> fprop(data *d);
  double              activate(Neuron *n, std::vector<double> weights); // dot product
  double              transfer(double x);                               // sigmoid
  double              transferDerivative(double x);                     // sigmoid derivative (backprop)
  void                bprop(data *d);
  void                updateWeights(data *d);
  int                 predict(data *d); // returns index of max value in ouptut array
  void                train(int epochs, double min_error = 0.0);
  double              test();
  double              validate();
};

#endif