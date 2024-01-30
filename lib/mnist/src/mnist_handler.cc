#include "mnist_handler.hh"
#include "data.hh"
#include <time.h>
#include <random>

mnist::mnist()
{
  data_array              = new std::vector<data *>();
  training_data           = new std::vector<data *>();
  testing_data            = new std::vector<data *>();
  validation_data         = new std::vector<data *>();
  num_classes             = 0;
  get_feature_vector_size = 0;
}

std::vector<data *> *mnist::get_training_data() { return training_data; }

std::vector<data *> *mnist::get_testing_data() { return testing_data; }

std::vector<data *> *mnist::get_validation_data() { return validation_data; }

mnist::~mnist()
{
  delete data_array;
  delete training_data;
  delete testing_data;
  delete validation_data;
}

void mnist::read_feature_vector(std::string path)
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
      if(fread(bytes, sizeof(bytes), 1, file)) { header[i] = convert_to_little_endian(bytes); }
    }
  printf("\33[2K\r");
  printf("\rDone getting input file header ");

  int magic      = header[0];
  int num_images = header[1];
  int num_rows   = header[2];
  int num_cols   = header[3];
  int image_size = num_cols * num_rows;
  printf("\33[2K\r");
  fflush(stdout);

  for(int i = 0; i < num_images; i++)
    {
      data   *d = new data();
      uint8_t element[1];
      printf("\rReading images: [ %d/%d ]", i + 1, num_images);
      fflush(stdout);
      for(int j = 0; j < image_size; j++)
        {
          if(fread(element, sizeof(element), 1, file)) { d->append_to_feature_vector(element[0]); }
          else
            {
              printf("Error reading image\n");
              exit(1);
            }
        }
      data_array->push_back(d);
    }
  printf("\33[2K\r");
  printf("\rDone getting images.\n");
  fflush(stdout);
}

void mnist::read_feature_labels(std::string path)
{
  uint32_t      header[2];
  unsigned char bytes[4];
  FILE         *file = fopen(path.c_str(), "rb");

  if(file == NULL)
    {
      printf("Error opening file\n");
      exit(1);
    }

  for(int i = 0; i < 2; i++)
    {
      if(fread(bytes, sizeof(bytes), 1, file)) { header[i] = convert_to_little_endian(bytes); }
    }

  int magic     = header[0];
  int num_items = header[1];

  printf("\33[2K\r");
  printf("\rDone getting lable file header");
  fflush(stdout);

  for(int i = 0; i < num_items; i++)
    {
      uint8_t element[1];

      if(fread(element, sizeof(element), 1, file))
        {
          printf("\rReading items: [ %d/%d ]", i + 1, num_items);
          fflush(stdout);
          data_array->at(i)->set_label(element[0]);
        }
    }
  printf("\33[2K\r");
  printf("\rDone getting lables.\n");
  fflush(stdout);
}

void mnist::fill_random(std::vector<data *> *vec, int num_samples, std::unordered_set<int> *used_indexes)
{
  int                                    count = 0;
  std::random_device                     rd;
  std::mt19937                           mt(rd());
  std::uniform_real_distribution<double> dist(0, data_array->size());

  while(count < num_samples)
    {
      int rnd_index = dist(mt);
      if(used_indexes->find(rnd_index) == used_indexes->end())
        {
          vec->push_back(data_array->at(rnd_index));
          printf("\33[2K\r");
          printf("Filling random: [ %d/%d ] used indexes count [%d]", count + 1, num_samples,
                 (int)used_indexes->size());
          fflush(stdout);
          used_indexes->insert(rnd_index);
          count++;
        }
      else
        {
          printf("\rFilling random: [ %d/%d ] used indexes count [%d]", count + 1, num_samples,
                 (int)used_indexes->size());
          printf(" : skipped [%d / %d] - [%d]  ", rnd_index, count + 1, num_samples, (int)used_indexes->size());
          continue;
        }
    }
}

void mnist::split_data()
{
  int num_training = data_array->size() * TRAIN_SET_PERCENT;
  int num_testing  = data_array->size() * TEST_SET_PERCENT;
  int num_valid    = data_array->size() * VALIDATION_SET_PERCENT;

  std::unordered_set<int> used_indexes;

  fill_random(validation_data, num_valid, &used_indexes);
  fill_random(training_data, num_training, &used_indexes);
  fill_random(testing_data, num_testing, &used_indexes);

  printf("\33[2K\r");
  printf("\rDone splitting data.");
  printf("\nTraining size: %zu\nTesting size: %zu\nValidation size: %zu\n", training_data->size(), testing_data->size(),
         validation_data->size());
}

void mnist::count_classes()
{
  int count = 0;
  for(int i = 0; i < data_array->size(); i++)
    {
      if(class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
          printf("\rCounting classes: [ %d/%zu ]", i + 1, data_array->size());
          fflush(stdout);
          class_map[data_array->at(i)->get_label()] = count;
          data_array->at(i)->set_enumerated_label(count);
          count++;
        }
    }
}

uint32_t mnist::convert_to_little_endian(const unsigned char *bytes)
{
  return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}
