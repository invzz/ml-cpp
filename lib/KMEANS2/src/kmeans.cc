#include "kmeans.hh"
#include <random>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>

template <typename T> T random(T range_from, T range_to)
{
  std::random_device               rand_dev;
  std::mt19937                     generator(rand_dev());
  std::uniform_int_distribution<T> distr(range_from, range_to);
  return distr(generator);
}

kmeans::kmeans(int k)
{
  //   training_data    = new std::vector<data *>();
  //   test_data        = new std::vector<data *>();
  //   validation_data  = new std::vector<data *>();
  clusters     = new std::vector<cluster_t *>();
  used_indexes = new std::unordered_set<int>();
  n_clusters   = k;
}

kmeans::~kmeans()
{
  for(auto cluster : *clusters) { delete cluster; }
  delete clusters;
  delete used_indexes;
}

void kmeans::init_clusters()
{
  for(int i = 0; i < n_clusters; i++)
    {
      int index = random((int)0, (int)(training_data->size() - 1));
      while(used_indexes->find(index) != used_indexes->end())
        {
          index = random((int)0, (int)(training_data->size() - 1));
        }
      used_indexes->insert(index);
      cluster_t *new_cluster = new cluster_t(training_data->at(index));
      clusters->push_back(new_cluster);
    }
}

void kmeans::init_clusters_for_each_class()
{
  std::unordered_set<int> used_classes;
  for(int i = 0; i < training_data->size(); i++)
    {
      if(used_classes.find(training_data->at(i)->get_label()) == used_classes.end())
        {
          used_classes.insert(training_data->at(i)->get_label());
          cluster_t *new_cluster = new cluster_t(training_data->at(i));
          clusters->push_back(new_cluster);
        }
    }
}

void kmeans::train()
{
  int                       count               = 0;
  std::vector<data *>      &local_training_data = *training_data;
  std::vector<cluster_t *> &local_clusters      = *clusters;
  std::atomic<int>          count_atomic(0);

  std::vector<std::thread> threads;
  for(int t = 0; t < KNN_NUM_OF_THREADS; ++t)
    {
      threads.emplace_back([&]() {
        int local_count = 0;
        for(int idx = t; idx < local_training_data.size(); idx += KNN_NUM_OF_THREADS)
          {
            auto   point        = local_training_data[idx];
            double min_distance = std::numeric_limits<double>::max();
            int    best_cluster = 0;
            for(int i = 0; i < local_clusters.size(); ++i)
              {
                double distance = euclidean_distance(local_clusters[i]->centroid, point);
                if(distance < min_distance)
                  {
                    min_distance = distance;
                    best_cluster = i;
                  }
              }
            {
              std::lock_guard<std::mutex> lock(mutex);
              local_clusters[best_cluster]->add_data_to_cluster(point);
              used_indexes->insert(point->get_label());
              local_count++;
            }
          }
        count_atomic += local_count;
        std::cout << "\rTraining: [ " << count_atomic << "/" << local_training_data.size()
                  << " ] => num clusters : " << local_clusters.size();
        fflush(stdout);
      });
    }

  for(auto &thread : threads) { thread.join(); }

  count = count_atomic.load();

  printf("\33[2K\r");
}

double kmeans::euclidean_distance(std::vector<double> *a, data *b)
{
  double distance = 0;
  for(int i = 0; i < a->size(); i++) { distance += pow(a->at(i) - b->get_feature_vector()->at(i), 2); }
  return sqrt(distance);
}

double kmeans::validate()
{
  printf("\33[2K\r");

  std::atomic<double> min_distance(std::numeric_limits<double>::max());
  std::atomic<int>    best_cluster(0);
  double              num_correct(0.0);
  std::atomic<int>    count(0);

  const int num_threads = KNN_NUM_OF_THREADS;
  const int chunk_size  = validation_data->size() / num_threads;

  std::vector<std::thread> threads;

  for(int t = 0; t < num_threads; ++t)
    {
      threads.emplace_back([&, t]() {
        int start = t * chunk_size;
        int end   = (t == num_threads - 1) ? validation_data->size() : (t + 1) * chunk_size;

        for(int idx = start; idx < end; ++idx)
          {
            printf("\rValidating: [ %d/%d ]", idx, (int)validation_data->size());
            fflush(stdout);
            auto query_point = (*validation_data)[idx];

            double local_min_distance = std::numeric_limits<double>::max();
            int    local_best_cluster = 0;

            for(int i = 0; i < clusters->size(); ++i)
              {
                double distance = euclidean_distance(clusters->at(i)->centroid, query_point);
                if(distance < local_min_distance)
                  {
                    local_min_distance = distance;
                    local_best_cluster = i;
                  }
              }

            double old_min_distance = min_distance.load();
            while(local_min_distance < old_min_distance &&
                  !min_distance.compare_exchange_weak(old_min_distance, local_min_distance))
              {}

            int old_best_cluster = best_cluster.load();
            while(local_best_cluster < old_best_cluster &&
                  !best_cluster.compare_exchange_weak(old_best_cluster, local_best_cluster))
              {}

            if(clusters->at(local_best_cluster)->most_freq_class == query_point->get_label()) { num_correct += 1.0; }

            count.fetch_add(1, std::memory_order_relaxed);
          }
      });
    }

  for(auto &thread : threads) { thread.join(); }

  printf(" :: %f", num_correct / static_cast<double>(validation_data->size()));
  return 100.0 * num_correct / static_cast<double>(validation_data->size());
}

double kmeans::test()
{
  std::atomic<double> min_distance(std::numeric_limits<double>::max());
  std::atomic<int>    best_cluster(0);
  double              num_correct(0.0);
  std::atomic<int>    count(0);

  const int num_threads = KNN_NUM_OF_THREADS;
  const int chunk_size  = test_data->size() / num_threads;

  std::vector<std::thread> threads;

  for(int t = 0; t < num_threads; ++t)
    {
      threads.emplace_back([&, t]() {
        int start = t * chunk_size;
        int end   = (t == num_threads - 1) ? test_data->size() : (t + 1) * chunk_size;
        printf("\33[2K\r");
        for(int idx = start; idx < end; ++idx)
          {
            printf("\rTesting: [ %d/%d ]", idx, (int)test_data->size());
            fflush(stdout);
            auto query_point = (*test_data)[idx];

            double local_min_distance = std::numeric_limits<double>::max();
            int    local_best_cluster = 0;

            for(int i = 0; i < clusters->size(); ++i)
              {
                double distance = euclidean_distance(clusters->at(i)->centroid, query_point);
                if(distance < local_min_distance)
                  {
                    local_min_distance = distance;
                    local_best_cluster = i;
                  }
              }

            double old_min_distance = min_distance.load();
            while(local_min_distance < old_min_distance &&
                  !min_distance.compare_exchange_weak(old_min_distance, local_min_distance))
              {}

            int old_best_cluster = best_cluster.load();
            while(local_best_cluster < old_best_cluster &&
                  !best_cluster.compare_exchange_weak(old_best_cluster, local_best_cluster))
              {}

            if(clusters->at(local_best_cluster)->most_freq_class == query_point->get_label()) { num_correct += 1.0; }

            count.fetch_add(1, std::memory_order_relaxed);
          }
      });
    }

  for(auto &thread : threads) { thread.join(); }

  printf(" :: %f\n", num_correct / static_cast<double>(test_data->size()));
  return 100.0 * num_correct / static_cast<double>(test_data->size());
}