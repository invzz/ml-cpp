#ifndef _iris_H_
#define _iris_H_
#include "data.hh"
#include "stdint.h"
#include "stdio.h"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

class iris
{
  std::vector<data *> *data_array;      // all of the data from the source
  std::vector<data *> *training_data;   // training data
  std::vector<data *> *testing_data;    // testing data
  std::vector<data *> *validation_data; // validation data

  int num_classes;
  int feature_vector_size;

  std::map<std::string, int> class_map; // maps the class to an integer

  public:
  iris();
  ~iris();
  void read_csv(std::string path, std::string delimiter = ",");

  void split_data();
  void fill();
  int  get_class_counts();
  int count_classes();
  void normalize();

  std::vector<data *> *get_training_data();
  std::vector<data *> *get_testing_data();
  std::vector<data *> *get_validation_data();

  private:
  void fill_random(std::vector<data *> *vec, int num_samples, std::unordered_set<int> *used_indexes);
};
#endif