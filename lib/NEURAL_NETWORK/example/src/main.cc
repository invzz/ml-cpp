#include "stdio.h"
#include "mnist_handler.hh"
#include "kmeans.hh"
#include "common.hh"

int main(int argc, char *argv[])
{
  mnist *m = new mnist();
  m->read_feature_vector(RES_DIR "/mnist/train-images.idx3-ubyte");
  m->read_feature_labels(RES_DIR "/mnist/train-labels.idx1-ubyte");
  m->split_data();
  m->count_classes();
  double performance      = 0.0;
  double best_performance = 0.0;
  int    best_k           = 1;
  int    training_size    = m->get_training_data()->size();
  for(int k = 1000; k < 2000; k += 10)
    {
      kmeans *km = new kmeans(k);
      km->set_training_data(m->get_training_data());
      km->set_test_data(m->get_testing_data());
      km->set_validation_data(m->get_validation_data());
      km->init_clusters();
      km->train();
      performance = km->validate();
      printf("- K: [ %d ], Performance: [ %.2f %% ]", k, performance);
      if(performance > best_performance)
        {
          best_performance = performance;
          best_k           = k;
        }
    }
  kmeans *km = new kmeans(best_k);
  km->set_training_data(m->get_training_data());
  km->set_test_data(m->get_testing_data());
  km->set_validation_data(m->get_validation_data());
  km->init_clusters();
  performance = km->test();
  printf("- K: [ %d ], Performance: [ %.2f %% ]", best_k, performance);

  return 0;
}