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

void iris::fill_random(std::vector<data *> *vec, int num_samples, std::unordered_set<int> *used_indexes)
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
          printf(" : [ %d ] skipped [%d / %d] - [%d]  ", rnd_index, count + 1, num_samples, (int)used_indexes->size());
          continue;
        }
    }
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
void iris::fill()
{
  training_data->clear();
  for(int i = 0; i < data_array->size(); i++) { training_data->push_back(data_array->at(i)); }
}
int iris::count_classes()
{
  // todo  return 0;
  return 0;
}

void iris::normalize()
{
  std::vector<double> mins, maxs;

  data *d = data_array->at(0);

  for(auto val : *d->get_feature_vector())
    {
      mins.push_back(val);
      maxs.push_back(val);
    }

  for(int i = 1; i < data_array->size(); i++)
    {
      d = data_array->at(i);
      for(int j = 0; j < d->get_feature_vector()->size(); j++)
        {
          double value = d->get_feature_vector()->at(j);
          if(value < mins[j]) { mins[j] = value; }
          if(value > maxs[j]) { maxs[j] = value; }
        }
    }

  for(int i = 0; i < data_array->size(); i++)
    {
      d = data_array->at(i);
      d->set_feature_vector(new std::vector<double>());
      d->set_class_vector(num_classes);
      for(int j = 0; j < d->get_feature_vector()->size(); j++)
        {
          if(maxs[j] - mins[j] == 0) { d->append_to_feature_vector(0.0); }
          else { d->append_to_feature_vector((d->get_feature_vector()->at(j) - mins[j]) / (maxs[j] - mins[j])); }
        }
    }

  // for(int i = 0; i < data_array->size(); i++)
  //   {
  //     std::vector<double> *fv = data_array->at(i)->get_NormalizedFeatureVector();
  //     double               sum = 0;
  //     for(int j = 0; j < fv->size(); j++) { sum += fv->at(j); }
  //     for(int j = 0; j < fv->size(); j++) { fv->at(j) = fv->at(j) / sum; }
  //   }
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
      d->set_feature_vector(new std::vector<double>());
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
  feature_vector_size = data_array->at(0)->get_NormalizedFeatureVector()->size();
}

int iris::get_class_counts() { return num_classes; }