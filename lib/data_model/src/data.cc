
#include "data.hh"

data::data()
{
  feature_vector = new std::vector<uint8_t>();
  label          = 0;
  enum_label     = 0;
}

data::~data() { delete feature_vector; }

void data::append_to_feature_vector(uint8_t value) { feature_vector->push_back(value); }
void data::append_to_feature_vector(double value) { NormalizedFeatureVector->push_back(value); }

void data::set_label(uint8_t value) { label = value; }
void data::set_enumerated_label(int value) { enum_label = value; }
void data::set_distance(double value) { distance = value; }
void data::set_class_vector(int count)
{
  class_vector = new std::vector<int>();
  for(int i = 0; i < count; i++)
    {
      if(i == label)
        class_vector->push_back(1);
      else
        class_vector->push_back(0);
    }
}
void data::set_feature_vector(std::vector<uint8_t> *v) { feature_vector = v; }
void data::set_NormalizedFeatureVector(std::vector<double> *v) { NormalizedFeatureVector = v; }

int                   data::get_feature_vector_size() { return feature_vector->size(); }
uint8_t               data::get_label() { return label; }
uint8_t               data::get_enumerated_label() { return enum_label; }
std::vector<uint8_t> *data::get_feature_vector() { return feature_vector; }
std::vector<int>      data::get_class_vector() { return *class_vector; }
std::vector<double>  *data::get_NormalizedFeatureVector() { return NormalizedFeatureVector; }

double data::get_distance() { return distance; }

void data::print_ascii_img()
{
  printf("-------------------------\n");
  for(int j = 0; j < feature_vector->size(); j++)
    {
      if(j > 0 && j % 28 == 0)
        {
          printf("\n");
          fflush(stdout);
        }
      printf("%*s", 2, feature_vector->at(j) > 0 ? "#" : " ");
      fflush(stdout);
    }
  printf("\n-------------------------\n");
}
