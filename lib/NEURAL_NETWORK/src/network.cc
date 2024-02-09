#include "network.hh"
#include "layer.hh"
#include "iris_handler.hh"
#include <numeric>

Network::Network(std::vector<int> hiddenLayerSpec, int inputSize, int numClasses, double learningRate)
{
  for(int i = 0; i < hiddenLayerSpec.size(); i++)
    {
      if(i == 0)
        Layers.push_back(new Layer(inputSize, hiddenLayerSpec.at(i)));
      else
        Layers.push_back(new Layer(Layers.at(i - 1)->neurons.size(), hiddenLayerSpec.at(i)));
    }
  Layers.push_back(new Layer(Layers.at(Layers.size() - 1)->neurons.size(), numClasses));
  this->learningRate = learningRate;
}

Network::~Network()
{
  // for(auto layer : Layers) { delete layer; }
}

double Network::activate(Neuron *n, std::vector<double> input)
{
  double activation = n->weights.back(); // bias term
  for(int i = 0; i < n->weights.size() - 1; i++) { activation += n->weights[i] * input[i]; }
  return activation;
}

double Network::transfer(double activation) { return 1.0 / (1.0 + exp(-activation)); } // sigmoid

double Network::transferDerivative(double output) { return output * (1 - output); } // sigmoid derivative (backprop)

std::vector<double> Network::fprop(data *d)
{
  std::vector<double> inputs = *d->get_NormalizedFeatureVector();

  for(int i = 0; i < Layers.size(); i++)
    {
      Layer              *layer = Layers.at(i);
      std::vector<double> newInputs;
      for(Neuron *n : layer->neurons)
        {
          double activation = this->activate(n, inputs);
          n->output         = this->transfer(activation);
          newInputs.push_back(n->output);
        }
      inputs = newInputs;
    }
  return inputs; // output vector of the output layer (last layer)
}

void Network::bprop(data *d)
{
  for(int i = Layers.size() - 1; i >= 0; i--)
    {
      std::vector<double> errors;
      Layer              *l = Layers.at(i);
      if(i != Layers.size() - 1)
        {
          // hidden layer
          for(int j = 0; j < l->neurons.size(); j++)
            { // calculate error for each neuron
              double error = 0.0;
              for(auto *n : Layers.at(i + 1)->neurons) { error += (n->weights.at(j) * n->delta); }
              errors.push_back(error);
            }
        }
      else
        {
          // output layer
          for(int j = 0; j < l->neurons.size(); j++)
            {
              // calculate error for each neuron
              auto n = l->neurons.at(j);
              errors.push_back(d->get_class_vector().at(j) - n->output); // error = expected - actual
            }
        }
      // update deltas
      for(int j = 0; j < l->neurons.size(); j++)
        {
          // calculate error for each neuron
          auto n   = l->neurons.at(j);
          n->delta = errors.at(j) * transferDerivative(n->output); // gradient / derivative
        }
    }
}

void Network::updateWeights(data *data)
{
  std::vector<double> inputs = *data->get_NormalizedFeatureVector();
  for(int i = 0; i < Layers.size(); i++)
    {
      if(i != 0)
        {
          for(Neuron *n : Layers.at(i - 1)->neurons) { inputs.push_back(n->output); }
        }
      for(Neuron *n : Layers.at(i)->neurons)
        {
          for(int j = 0; j < inputs.size(); j++) { n->weights.at(j) += this->learningRate * n->delta * inputs.at(j); }
          n->weights.back() += this->learningRate * n->delta;
        }
      inputs.clear();
    }
}
int Network::predict(data *d)
{
  std::vector<double> outputs = fprop(d);
  return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int epochs, double min_error)
{
  for(int i = 0; i < epochs; i++)
    {
      double sum_error = 0.0;
      for(data *d : *training_data)
        {
          std::vector<double> outputs      = fprop(d);
          std::vector<int>    expected     = d->get_class_vector();
          double              tempErrorSum = 0;
          for(int j = 0; j < expected.size(); j++) { tempErrorSum += pow(expected.at(j) - outputs.at(j), 2); }
          sum_error += tempErrorSum;
          bprop(d);
          updateWeights(d);
        }
      printf("\33[2K\r");
      printf("epoch: [ %d ] :: squared error: [ %.4f ]", i, sum_error);
      fflush(stdout);
      if(sum_error < min_error) { break; }
    }
}

double Network::test()
{
  double correct = 0;
  double count   = 0;
  for(data *d : *get_test_data())
    {
      count++;
      int prediction = predict(d);
      if(d->get_class_vector().at(prediction) == 1) { correct++; }
    }
  return (double)correct / count;
}

double Network::validate()
{
  double correct = 0;
  double count   = 0;
  for(data *d : *get_validation_data())
    {
      count++;
      int prediction = predict(d);
      if(d->get_class_vector().at(prediction) == 1) { correct++; }
    }
  return (double)correct / count;
}