#include "common.hh"

// setters
void common_data::set_training_data(std::vector<data *> *data) { training_data = data; }
void common_data::set_test_data(std::vector<data *> *data) { test_data = data; }
void common_data::set_validation_data(std::vector<data *> *data) { validation_data = data; }

// getters
std::vector<data *> *common_data::get_training_data() { return training_data; }
std::vector<data *> *common_data::get_test_data() { return test_data; }
std::vector<data *> *common_data::get_validation_data() { return validation_data; }