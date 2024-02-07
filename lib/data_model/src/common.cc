#include "common.hh"

void common_data::set_training_data(std::vector<data *> *data) { training_data = data; }

void common_data::set_test_data(std::vector<data *> *data) { test_data = data; }

void common_data::set_validation_data(std::vector<data *> *data) { validation_data = data; }