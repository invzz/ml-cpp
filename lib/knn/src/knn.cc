#include "knn.hh"
#include <cmath>
#include <map>
#include <iostream>
#include <algorithm>
#include <thread>
#include "stdint.h"
#include "mnist_handler.hh"
#include "thread_queue.hh"

double euclidean_distance(data *query_point, data *input)
{
  double distance = 0.0;
  if(query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
      std::cout << "\nError: feature vector sizes do not match" << std::endl;
      return -1;
    }
  for(unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
      distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
  return sqrt(distance);
}
knn::knn(int value) { k = value; }
knn::knn(int k_val, std::vector<data *> *training, std::vector<data *> *test, std::vector<data *> *validation)
{
  set_k(k_val);
  set_training_data(training);
  set_test_data(test);
  set_validation_data(validation);
}
knn::~knn()
{ // Nothing to do.
}
knn::knn() { k = 0; }

void knn::find_k_nearest_neighbors(data *d)
{
  neighbors           = new std::vector<data *>(); // Neighbourhood of the query point
  double min          = std::numeric_limits<double>::max();
  double previous_min = std::numeric_limits<double>::max();
  int    index        = 0;
  // Function to calculate distance, assuming you have a function named calculate_distance
  auto _distance = [](data *d1, data *d2) {
    // Implement your distance calculation logic here
    // This is just a placeholder; replace it with your actual implementation
    return euclidean_distance(d1, d2);
  };

  // Number of threads
  const unsigned num_threads = 20;
  // Split the loop into chunks for each thread
  const unsigned chunk_size = training_data->size() / num_threads;
  // Calculate the chunk size and the number of remaining elements
  const unsigned remaining_elements = training_data->size() % num_threads;
  // Vector to store threads
  std::vector<std::thread> threads;
  // Mutex for protecting vector modifications
  std::mutex vector_mutex;

  unsigned start = 0;

  for(unsigned t = 0; t < num_threads; ++t)
    {
      unsigned end = start + chunk_size + (t < remaining_elements ? 1 : 0);

      threads.emplace_back([&, t, start, end]() {
        for(unsigned curr_k = 0; curr_k < k; ++curr_k)
          {
            if(curr_k == 0)
              {
                for(unsigned i = start; i < end; ++i)
                  {
                    double distance = calculate_distance(d, training_data->at(i));
                    training_data->at(i)->set_distance(distance);

                    // Lock the vector before modification
                    std::lock_guard<std::mutex> lock(vector_mutex);
                    if(distance < min)
                      {
                        min   = distance;
                        index = i;
                      }
                  }

                // Lock the vector before modification
                std::lock_guard<std::mutex> lock(vector_mutex);
                neighbors->push_back(training_data->at(index));
                previous_min = min;
                min          = std::numeric_limits<double>::max();
              }
            else
              {
                for(unsigned i = start; i < end; ++i)
                  {
                    double distance = training_data->at(i)->get_distance();

                    // Lock the vector before modification
                    std::lock_guard<std::mutex> lock(vector_mutex);
                    if(distance < min && distance > previous_min)
                      {
                        min   = distance;
                        index = i;
                      }
                  }

                // Lock the vector before modification
                std::lock_guard<std::mutex> lock(vector_mutex);
                neighbors->push_back(training_data->at(index));
                previous_min = min;
                min          = std::numeric_limits<double>::max();
              }
          }
      });

      start = end;
    }

  // Wait for all threads to finish
  for(auto &thread : threads) { thread.join(); }

  return;
}

void knn::set_k(int val) { k = val; }

void knn::set_training_data(std::vector<data *> *data) { training_data = data; }

void knn::set_test_data(std::vector<data *> *data) { test_data = data; }

void knn::set_validation_data(std::vector<data *> *data) { validation_data = data; }

int knn::predict()
{
  std::map<uint8_t, int> klass_freq;
  for(unsigned i = 0; i < neighbors->size(); i++)
    {
      if(klass_freq.find(neighbors->at(i)->get_label()) == klass_freq.end())
        {
          klass_freq[neighbors->at(i)->get_label()] = 1;
        }
      else { klass_freq[neighbors->at(i)->get_label()] += 1; }
    }
  int     max_freq = 0;
  uint8_t klass    = 0;
  for(auto it : klass_freq)
    {
      if(it.second > max_freq)
        {
          max_freq = it.second;
          klass    = it.first;
        }
    }
  neighbors->clear();
  return klass;
}

double knn::calculate_distance(data *query_point, data *input)
{
  double distance = 0.0;
  return euclidean_distance(query_point, input);
}

double knn::compute_performance(std::vector<data *> *d)
{
  int correct = 0.0;
  int parsed  = 0.0;
  for(data *qp : *d)
    {
      find_k_nearest_neighbors(qp);
      int prediction = predict();
      if(prediction == qp->get_label()) { correct += 1; }
      parsed += 1;
      printf("\33[2K\r");
      printf("\rPrediction: [ %d ] :: Actual: [ %d ] :: ", prediction, qp->get_label());
      printf("Accuracy: [ %.3f %% ] :: ", ((double)correct / (double)parsed) * 100.0);
      printf("Progress: [ %.3f %% ] ", ((double)parsed / (double)d->size()) * 100.0);
      fflush(stdout);
    }
  printf("\nValidation complete :: Accuracy: [ %.3f %% ] k = [%d]", ((double)correct / (double)d->size()) * 100.0, k);
  return ((double)correct / (double)d->size());
}

double knn::compute_test_performance() { return compute_performance(test_data); }
double knn::compute_validation_performance() { return compute_performance(validation_data); }