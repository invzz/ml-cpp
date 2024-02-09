#ifndef _NEURON_H_
#define _NEURON_H_

#include <cmath>
#include <vector>

class Neuron
{
  public:
  double              output;
  double              delta;
  std::vector<double> weights;

  Neuron(int prevLayerSize, int currentLayerSize);
  ~Neuron();

  void initializeWeights(int prevLayerSize);
};

#endif