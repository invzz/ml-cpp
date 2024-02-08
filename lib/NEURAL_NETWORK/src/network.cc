#include "network.hh"
#include "layer.hh"
#include "iris_handler.hh"
#include <numeric>

Network::Network(std::vector<int> hiddenLayerSpec, int inputSize, int numClasses, double learningRate)
{
  // Create input layer
  int    prev       = inputSize;
  Layer *InputLayer = new Layer(prev, hiddenLayerSpec[0]);
  Layers.push_back(InputLayer);

  // Create hidden layers
  for(int i = 1; i < hiddenLayerSpec.size() - 1; i++)
    {
      int  prev        = Layers.at(i - 1)->neurons->size();
      auto HiddenLayer = new Layer(prev, hiddenLayerSpec[i]);
      Layers.push_back(HiddenLayer);
    }

  // Create output layer
  prev             = Layers.at(Layers.size() - 1)->neurons->size();
  auto Outputlayer = new Layer(prev, numClasses);
  Layers.push_back(Outputlayer);
}

Network::~Network()
{
  for(auto layer : Layers) { delete layer; }
}

double Network::activate(std::vector<double> *inputs, std::vector<double> *weights)
{
  double activation = weights->back(); // bias term
  for(int i = 0; i < weights->size() - 1; i++) { activation += inputs->at(i) * weights->at(i); }
  return activation;
}

double Network::transfer(double activation) { return 1.0 / (1.0 + exp(-activation)); } // sigmoid

double Network::transferDerivative(double output) { return output * (1 - output); } // sigmoid derivative (backprop)

std::vector<double> Network::fprop(data *d)
{
  std::vector<double> inputs = *d->get_NormalizedFeatureVector();

  for(int i = 0; i < Layers.size(); i++)
    {
      std::vector<double> newInputs;
      for(auto n : *Layers.at(i)->neurons)
        {
          double activation = activate(&inputs, n->weights);
          n->output         = transfer(activation);
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
          for(int j = 0; j < l->neurons->size(); j++)
            { // calculate error for each neuron
              double error = 0.0;
              for(auto n : *Layers.at(i + 1)->neurons) { error += n->weights->at(j) * n->delta; }
              errors.push_back(error);
            }
        }
      else
        {
          // output layer
          for(int j = 0; j < l->neurons->size(); j++)
            {
              // calculate error for each neuron
              auto n = l->neurons->at(j);
              errors.push_back(d->get_class_vector()->at(j) - n->output); // error = expected - actual
            }
        }
      // update deltas
      for(int j = 0; j < l->neurons->size(); j++)
        {
          // calculate error for each neuron
          auto n   = l->neurons->at(j);
          n->delta = errors.at(j) * transferDerivative(n->output); // gradient / derivative
        }
    }
}

void Network::updateWeights(data *d)
{
  std::vector<double> inputs = *d->get_NormalizedFeatureVector();
  for(int i = 0; i < Layers.size(); i++)
    {
      if(i != 0)

        {
          for(auto n : *Layers.at(i - 1)->neurons) { inputs.push_back(n->output); }
        }
      for(auto n : *Layers.at(i)->neurons)
        {
          for(int j = 0; j < inputs.size(); j++)
            {
              // update weights of the neuron
              n->weights->at(j) += learningRate * n->delta * inputs.at(j);
            }
          n->weights->back() += learningRate * n->delta; // bias
        }
    }
  inputs.clear();
}

int Network::predict(data *d)
{
  std::vector<double> outputs = fprop(d);
  return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int epochs)
{
  for(int i = 0; i < epochs; i++)
    {
      double sum_error = 0;
      for(data *d : *get_training_data())
        {
          std::vector<double> outputs      = fprop(d);
          std::vector<int>    expected     = *d->get_class_vector();
          double              tempErrorSum = 0;
          for(int j = 0; j < expected.size(); j++) { tempErrorSum += pow(expected.at(j) - outputs.at(j), 2); }
          sum_error += tempErrorSum;
          bprop(d);
          updateWeights(d);
        }
      printf("epoch: %d, error: %.4f\n", i, sum_error);
    }
}

double Network::test()
{
  double correct = 0;
  double count   = 0;
  for(data *d : *get_testing_data())
    {
      count++;
      int prediction = predict(d);
      if(d->get_class_vector()->at(prediction) == 1) { correct++; }
    }
  return (double)correct / count;
}

void Network::validate()
{
  double correct = 0;
  double count   = 0;
  for(data *d : *get_validation_data())
    {
      count++;
      int prediction = predict(d);
      if(d->get_class_vector()->at(prediction) == 1) { correct++; }
    }
  printf("Validation accuracy: %.2f%%\n", (correct / count) * 100);
}