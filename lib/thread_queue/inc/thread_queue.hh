#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T> class ThreadSafeQueue
{
  std::mutex              mutex;
  std::condition_variable cond_var;
  std::queue<T>           queue;

  public:
  void push(T &&item)
  {
    {
      std::lock_guard lock(mutex);
      queue.push(item);
    }

    cond_var.notify_one();
  }

  T &front()
  {
    std::unique_lock lock(mutex);
    cond_var.wait(lock, [&] { return !queue.empty(); });
    return queue.front();
  }

  void pop()
  {
    std::lock_guard lock(mutex);
    queue.pop();
  }
};