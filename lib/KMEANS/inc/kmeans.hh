#ifndef __KMEANS_HH__
#define __KMEANS_HH__
#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include <mutex>
#include "data.hh"
#include "common.hh"

typedef struct cluster
{
  std::vector<double> *centroid;
  std::vector<data *> *cluster_points;
  std::map<int, int>   class_counts;
  int                  most_freq_class;
  cluster(data *init_point)
  {
    centroid       = new std::vector<double>;
    cluster_points = new std::vector<data *>();
    for(auto value : *(init_point->get_feature_vector())) { centroid->push_back(value); }
    cluster_points->push_back(init_point);
    class_counts[init_point->get_label()] = 1;
    most_freq_class                       = init_point->get_label();
  }
  void add_data_to_cluster(data *point)
  {
    fflush(stdout);
    int previous_size = cluster_points->size();
    cluster_points->push_back(point);
    for(int i = 0; i < centroid->size() - 1; i++)
      {
        double value = centroid->at(i);
        value *= previous_size;
        value += point->get_feature_vector()->at(i);
        value /= (double)cluster_points->size();
        centroid->at(i) = value;
      }

    if(class_counts.find(point->get_label()) == class_counts.end()) { class_counts[point->get_label()] = 1; }
    else { class_counts[point->get_label()]++; }
    set_most_freq_class();
  }
  void set_most_freq_class()
  {
    int best_class = 0;
    int freq       = 0;

    for(auto KV : class_counts)
      {
        if(KV.second > freq)
          {
            freq       = KV.second;
            best_class = KV.first;
          }
      }
    most_freq_class = best_class;
  }

} cluster_t;

class kmeans : public common_data
{
  int                       n_clusters;
  std::vector<cluster_t *> *clusters;
  std::unordered_set<int>  *used_indexes;

  public:
  kmeans(int n_clusters);
  ~kmeans();
  void   init_clusters();
  void   init_clusters_for_each_class();
  void   train();
  double euclidean_distance(std::vector<double> *a, data *b);
  double validate();
  double test();

  private:
  std::mutex mutex;
};

#endif