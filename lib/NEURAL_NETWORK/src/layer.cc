#include "layer.hh"

Layer::Layer(int prevSize, int currentSize)
{
    for(int i = 0; i < currentSize; i++) { neurons.push_back(new Neuron(prevSize, currentSize)); }
  this->currentSize = currentSize;
}