#ifndef _mnist_H_
#define _mnist_H_
#include "data.hh"
#include "stdint.h"
#include "stdio.h"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

class mnist
{
  std::vector<data *> *data_array;      // all of the data from the source
  std::vector<data *> *training_data;   // training data
  std::vector<data *> *testing_data;    // testing data
  std::vector<data *> *validation_data; // validation data

  int num_classes;
  int get_feature_vector_size;

  std::map<uint8_t, int> class_map; // maps the class to an integer

  public:
  mnist();
  ~mnist();

  void read_feature_vector(std::string path);
  void read_feature_labels(std::string path);

  void split_data();
  void count_classes();
  void fill();

  int get_class_counts();

  uint32_t convert_to_little_endian(const unsigned char *bytes);

  std::vector<data *> *get_training_data();
  std::vector<data *> *get_testing_data();
  std::vector<data *> *get_validation_data();

  private:
  void fill_random(std::vector<data *> *vec, int num_samples, std::unordered_set<int> *used_indexes);
};
#endif