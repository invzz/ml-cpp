#ifndef _LAYER_H_
#define _LAYER_H_

#include "neuron.hh"
#include <vector>

class Layer
{
  public:
  std::vector<Neuron *> *neurons;
  std::vector<double>   *outputs;
  int                    currentSize;
  Layer(int prevSize, int currentSize);
  ~Layer();
};

#endif