#include "main.hh"
#include <iostream>


int KNN_find_best_k(mnist *dh, int min_k, int max_k)
{
  double best_acc = 0.0;
  int    best_k   = 0;
  std::cout << "\nComputing Test performance: " << std::endl;
  for(int k = min_k; k <= max_k; k++)
    {
      knn *knn_classifier = new knn(k, dh->get_training_data(), dh->get_testing_data(), dh->get_validation_data());
      auto acc            = knn_classifier->compute_test_performance();
      if(acc > best_acc)
        {
          best_acc = acc;
          best_k   = k;
        }
    }
  std::cout << "[ Best k: " << best_k << "]"
            << "[Acc : " << best_acc * 100 << "]" << std::endl;
  std::cout << "Computing Validation performance: " << std::endl;
  knn *knn_classifier = new knn(best_k, dh->get_training_data(), dh->get_testing_data(), dh->get_validation_data());
  knn_classifier->compute_validation_performance();
  return best_k;
}

int main()
{
  std::cout << "number of threads: " << KNN_NUM_OF_THREADS << std::endl;
  mnist *dh = new mnist();
  dh->read_feature_vector(TRAINING_IMAGES_FILE);

  dh->read_feature_labels(TRAINING_LABELS_FILE);
  dh->split_data();
  dh->count_classes();
  auto best_acc = KNN_find_best_k(dh, 1, 1);
}
