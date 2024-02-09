#include "stdio.h"
#include "iris_handler.hh"
#include "network.hh"
#include "common.hh"

int main(int argc, char *argv[])
{
  iris *i = new iris();
  i->read_csv(RES_DIR "/iris/iris.data");
  i->split_data();

  std::vector<int> hiddenLayerSpec = {10};
  auto             inputSize       = i->get_training_data()->at(0)->get_NormalizedFeatureVector()->size();
  auto             numClasses      = i->get_class_counts();
  auto             learningRate    = .01;

  auto lamda = [&](double learningRate = .95, int epochs = 100) {
    Network *net = new Network(hiddenLayerSpec, inputSize, numClasses, learningRate);
    net->set_training_data(i->get_training_data());
    net->set_test_data(i->get_testing_data());
    net->set_validation_data(i->get_validation_data());
    net->train(epochs);
    net->validate();
    double val  = net->validate();
    double test = net->test();
    printf("\repochs -> [ %d ] Validation: [ %3.3f%% ] Test: [ %3.3f%% ]", epochs, val * 100, test * 100);
  };

  lamda(learningRate, 1000);
  printf("- Learning rate: %2.2f\n", learningRate);

  return 0;
}