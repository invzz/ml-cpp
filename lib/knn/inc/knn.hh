#ifndef _H_KNN_
#define _H_KNN_

#include "common.hh"
#include <mutex>
#include <vector>

class knn : public common_data
{
  int                  k;
  std::vector<data *> *neighbors;

  double compute_performance(std::vector<data *> *);

  public:
  std::mutex vector_mutex;
  knn(int);
  knn(int, std::vector<data *> *, std::vector<data *> *, std::vector<data *> *);
  knn();
  ~knn();

  void find_k_nearest_neighbors(data *data);
  void set_k(int val);

  int    predict();
  double calculate_distance(data *query_point, data *input);
  double compute_test_performance();
  double compute_validation_performance();
};

#endif