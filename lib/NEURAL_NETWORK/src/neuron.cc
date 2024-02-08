
#include "neuron.hh"
#include <random>
template <typename T> T random(T range_from, T range_to)
{
  std::random_device                                       dev;
  std::mt19937                                             rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(range_from, range_to); // distribution in range [1, 6]
  return dist6(rng);
}

Neuron::Neuron(int prevSize, int currentSize)
{
  initializeWeights(prevSize);
  output = 0.0;
  delta  = 0.0;
}

Neuron::~Neuron() { delete weights; }

void Neuron::initializeWeights(int prevSize)
{
  weights = new std::vector<double>();
  for(int i = 0; i < prevSize + 1; i++) // +1 for the bias
    {
      weights->push_back(random(-1.0f, 1.0f));
    }
}
