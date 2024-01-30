#ifndef _H_KNN_
#define _H_KNN_

#include "data.hh"

#include <vector>

class knn
{
  int                  k;
  std::vector<data *> *neighbors;
  std::vector<data *> *training_data;
  std::vector<data *> *test_data;
  std::vector<data *> *validation_data;

  double compute_performance(std::vector<data *> *);

  public:
  knn(int);
  knn(int, std::vector<data *> *, std::vector<data *> *, std::vector<data *> *);
  knn();
  ~knn();

  void find_k_nearest_neighbors(data *data);
  void set_k(int val);
  void set_training_data(std::vector<data *> *data);
  void set_test_data(std::vector<data *> *data);
  void set_validation_data(std::vector<data *> *data);

  int    predict();
  double calculate_distance(data *query_point, data *input);
  double compute_test_performance();
  double compute_validation_performance();
};

#endif