#ifndef NETWORK_H__
#define NETWORK_H__

#include "data.hh"
#include "neuron.hh"
#include "layer.hh"
#include "iris_handler.hh"

class Network : public iris
{
  public:
  std::vector<Layer *> Layers;
  double               learningRate;
  double               testPerfomance;

  Network(std::vector<int> hiddenLayerSpec, int, int, double);
  ~Network();

  std::vector<double> fprop(data *d);
  double              activate(std::vector<double> *inputs, std::vector<double> *weights); // dot product
  double              transfer(double x);                                                  // sigmoid
  double              transferDerivative(double x); // sigmoid derivative (backprop)
  void                bprop(data *d);
  void                updateWeights(data *d);
  int                 predict(data *d); // returns index of max value in ouptut array
  void                train(int epochs);
  double              test();
  void                validate();
};

#endif