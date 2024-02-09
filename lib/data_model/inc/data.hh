#ifndef _DATA_M_H_
#define _DATA_M_H_
#include "stdint.h"
#include "stdio.h"
#include <vector>
#include <unordered_set>

class data
{
  std::vector<uint8_t> *feature_vector;          // data
  std::vector<double>  *NormalizedFeatureVector; // data
  std::vector<int>     *class_vector;            // data

  uint8_t label;
  int     enum_label;
  double  distance;

  public:
  data();
  ~data();
  void set_feature_vector(std::vector<uint8_t> *);
  void set_NormalizedFeatureVector(std::vector<double> *v);

  void append_to_feature_vector(uint8_t);
  void append_to_feature_vector(double);

  void set_label(uint8_t);
  void set_enumerated_label(int);
  void set_distance(double);

  void set_class_vector(int count);

  std::vector<uint8_t> *get_feature_vector();
  std::vector<double>  *get_NormalizedFeatureVector();
  std::vector<int>      get_class_vector();

  uint8_t get_label();
  uint8_t get_enumerated_label();

  int    get_feature_vector_size();
  double get_distance();

  void print_ascii_img();
};

#endif