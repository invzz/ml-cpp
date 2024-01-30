#include "main.hh"
#ifndef TRAINING_IMAGES_FILE
#define TRAINING_DATA_FILE "train-images-idx3-ubyte"
#endif
#ifndef TRAINING_LABELS_FILE
#define TRAINING_LABELS_FILE "train-labels-idx3-ubyte"
#endif

int main()
{
  mnist *dh = new mnist();

  dh->read_feature_vector(TRAINING_IMAGES_FILE);
  dh->read_feature_labels(TRAINING_LABELS_FILE);
  dh->split_data();
  dh->count_classes();

  knn *knn_classifier = new knn(2, dh->get_training_data(), dh->get_testing_data(), dh->get_validation_data());
  knn_classifier->compute_validation_performance();

  return 0;
}
