
#include "data.hh"

data::data()
{
  feature_vector = new std::vector<uint8_t>();
  label          = 0;
  enum_label     = 0;
}

data::~data() { delete feature_vector; }

void data::set_feature_vector(std::vector<uint8_t> *v) { feature_vector = v; }

void data::append_to_feature_vector(uint8_t value) { feature_vector->push_back(value); }

void data::set_label(uint8_t value) { label = value; }

void data::set_enumerated_label(int value) { enum_label = value; }

void data::set_distance(double value) { distance = value; }

int data::get_feature_vector_size() { return feature_vector->size(); }

uint8_t data::get_label() { return label; }

uint8_t data::get_enumerated_label() { return enum_label; }

std::vector<uint8_t> *data::get_feature_vector() { return feature_vector; }

double data::get_distance() { return distance; }

