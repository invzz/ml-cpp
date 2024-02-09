#ifndef COMMON_HH
#define COMMON_HH

#include <vector>
#include "data.hh"

class common_data
{
  protected:
  std::vector<data *> *training_data;
  std::vector<data *> *test_data;
  std::vector<data *> *validation_data;

  public:
  // setters
  void set_training_data(std::vector<data *> *data);
  void set_test_data(std::vector<data *> *data);
  void set_validation_data(std::vector<data *> *data);
  
  // getters
  std::vector<data *> *get_training_data();
  std::vector<data *> *get_test_data();
  std::vector<data *> *get_validation_data();
};

#endif
