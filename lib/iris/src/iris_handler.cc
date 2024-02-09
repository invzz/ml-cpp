#include "iris_handler.hh"
#include "data.hh"
#include <time.h>
#include <random>
#include <numeric>
#include <fstream>

iris::iris()
{
  data_array          = new std::vector<data *>();
  training_data       = new std::vector<data *>();
  testing_data        = new std::vector<data *>();
  validation_data     = new std::vector<data *>();
  num_classes         = 0;
  feature_vector_size = 0;
}

std::vector<data *> *iris::get_training_data() { return training_data; }

std::vector<data *> *iris::get_testing_data() { return testing_data; }

std::vector<data *> *iris::get_validation_data() { return validation_data; }

iris::~iris()
{
  delete data_array;
  delete training_data;
  delete testing_data;
  delete validation_data;
}

void iris::split_data()
{
  int num_training = data_array->size() * TRAIN_SET_PERCENT;
  int num_testing  = data_array->size() * TEST_SET_PERCENT;
  int num_valid    = data_array->size() * VALIDATION_SET_PERCENT;

  // Shuffle the indices
  std::vector<int> indices(data_array->size());
  std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., data_array->size()-1
  std::random_device rd;
  std::mt19937       g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  // Assign data to training, testing, and validation sets
  for(int i = 0; i < num_training; ++i) training_data->push_back(data_array->at(indices[i]));

  for(int i = num_training; i < num_training + num_testing; ++i) testing_data->push_back(data_array->at(indices[i]));

  for(int i = num_training + num_testing; i < num_training + num_testing + num_valid; ++i)
    validation_data->push_back(data_array->at(indices[i]));

  printf("\33[2K\r");
  printf("\rDone splitting data.");
  printf("\nTraining size: %zu\nTesting size: %zu\nValidation size: %zu\n", training_data->size(), testing_data->size(),
         validation_data->size());
}

void iris::normalize()
{
  std::vector<double> mins, maxs;
  // fill min and max lists

  data *d = data_array->at(0);
  for(auto val : *d->get_NormalizedFeatureVector())
    {
      mins.push_back(val);
      maxs.push_back(val);
    }

  for(int i = 1; i < data_array->size(); i++)
    {
      d = data_array->at(i);
      for(int j = 0; j < d->get_NormalizedFeatureVector()->size(); j++)
        {
          double value = (double)d->get_NormalizedFeatureVector()->at(j);
          if(value < mins.at(j)) mins[j] = value;
          if(value > maxs.at(j)) maxs[j] = value;
        }
    }
  // normalize data array

  for(int i = 0; i < data_array->size(); i++)
    {
      data_array->at(i)->set_NormalizedFeatureVector(new std::vector<double>());
      data_array->at(i)->set_class_vector(num_classes);
      for(int j = 0; j < data_array->at(i)->get_NormalizedFeatureVector()->size(); j++)
        {
          if(maxs[j] - mins[j] == 0)
            data_array->at(i)->append_to_feature_vector(0.0);
          else
            data_array->at(i)->append_to_feature_vector(
              (double)(data_array->at(i)->get_NormalizedFeatureVector()->at(j) - mins[j]) / (maxs[j] - mins[j]));
        }
    }
}

void iris::read_csv(std::string path, std::string delimiter)
{
  num_classes = 0;
  std::ifstream datafile(path.c_str());
  std::string   line;
  while(std::getline(datafile, line))
    {
      if(line.length() == 0) continue;
      data *d = new data();
      d->set_NormalizedFeatureVector(new std::vector<double>());
      size_t      pos = 0;
      std::string token;
      while((pos = line.find(delimiter)) != std::string::npos)
        {
          token = line.substr(0, pos);
          d->append_to_feature_vector(std::stod(token));
          line.erase(0, pos + delimiter.length());
        }
      if(class_map.find(line) == class_map.end())
        {
          class_map[line] = num_classes;
          d->set_label(num_classes);
          num_classes++;
        }
      else { d->set_label(class_map[line]); }
      data_array->push_back(d);
    }
  for(auto d : *data_array) { d->set_class_vector(num_classes); }
  feature_vector_size = data_array->at(0)->get_NormalizedFeatureVector()->size();
}

int iris::get_class_counts() { return num_classes; }