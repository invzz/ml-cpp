#include "data_handler.hh"

data_handler::data_handler()
{
  data_array              = new std::vector<data *>();
  training_data           = new std::vector<data *>();
  testing_data            = new std::vector<data *>();
  validation_data         = new std::vector<data *>();
  num_classes             = 0;
  get_feature_vector_size = 0;
}

data_handler::~data_handler()
{
  delete data_array;
  delete training_data;
  delete testing_data;
  delete validation_data;
}

void data_handler::read_feature_vector(std::string path)
{
  uint32_t      header[4];
  unsigned char bytes[4];
  FILE         *file = fopen(path.c_str(), "rb");

  if(file == NULL)
    {
      printf("Error opening file\n");
      exit(1);
    }

  for(int i = 0; i < 4; i++)
    {
      if(fread(bytes, sizeof(bytes), 1, file)) {
        header[i] = convert_to_little_endian(bytes);
      }
    }
    printf("\nDone getting header");
    int image_size = header[2] * header[3];

}

void data_handler::read_feature_labels(std::string path) {}

void data_handler::split_data() {}

void data_handler::count_classes() {}

uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes) { return 0; }

std::vector<data *> *data_handler::get_training_data() { return nullptr; }

std::vector<data *> *data_handler::get_testing_data() { return nullptr; }

std::vector<data *> *data_handler::get_validation_data() { return nullptr; }
