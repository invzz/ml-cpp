#ifndef _DATA_HANDLER_H_
#define _DATA_HANDLER_H_
#include "data.hh"
#include "stdint.h"
#include "stdio.h"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

class data_handler
{
  const double TRAIN_SET_PERCENT      = 0.75;
  const double TEST_SET_PERCENT       = 0.20;
  const double VALIDATION_SET_PERCENT = 0.05;

  std::vector<data *> *data_array;      // all of the data before splitting
  std::vector<data *> *training_data;   // training data
  std::vector<data *> *testing_data;    // testing data
  std::vector<data *> *validation_data; // validation data

  int num_classes;
  int get_feature_vector_size;

  std::map<uint8_t, int> class_map; // maps the class to an integer

  public:
  data_handler();
  ~data_handler();

  void read_feature_vector(std::string path);
  void read_feature_labels(std::string path);

  void split_data();
  void count_classes();

  uint32_t convert_to_little_endian(const unsigned char *bytes);

  std::vector<data *> *get_training_data();
  std::vector<data *> *get_testing_data();
  std::vector<data *> *get_validation_data();

  private:
  void fill_random(std::vector<data *> *vec, int num_samples, std::unordered_set<int> *used_indexes);
};
#endif