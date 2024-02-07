#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "knn.hh"
#include "mnist_handler.hh"

std::vector<cv::Point> points;
int                    is_drawing = 0;
cv::Mat                img;
cv::Mat                img2;
cv::Mat                img3  = cv::Mat::zeros(cv::Size(280, 50), CV_8UC1);
int                    klass = 0;
mnist                 *m     = new mnist();

std::vector<uint8_t> *img_to_vector(cv::Mat img)
{
  std::vector<uint8_t> *vec = new std::vector<uint8_t>();
  for(int i = 0; i < img.rows; i++)
    {
      for(int j = 0; j < img.cols; j++) { vec->push_back(img.at<uchar>(i, j)); }
    }
  return vec;
}

void drawCircle(int event, int x, int y, int, void *param)
{
  if(event == cv::EVENT_LBUTTONDOWN)
    {
      img3 = cv::Mat::zeros(cv::Size(280, 50), CV_8UC1);
      cv::putText(img3, "predicting", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(118, 185, 0), 1);
      is_drawing = 1;
    }
  if(event == cv::EVENT_LBUTTONUP)
    {
      is_drawing = 0;
      img3       = cv::Mat::zeros(cv::Size(280, 50), CV_8UC1);
      cv::resize(img, img2, img2.size());
      std::vector<data *> *training_data = m->get_training_data();
      std::vector<data *> *testing_data  = new std::vector<data *>();
      knn                 *k             = new knn(5);
      testing_data->push_back(new data());
      auto img_vec = img_to_vector(img2);
      testing_data->at(0)->set_feature_vector(img_vec);
      k->set_test_data(testing_data);
      k->set_training_data(training_data);
      k->find_k_nearest_neighbors(testing_data->at(0));
      testing_data->at(0)->print_ascii_img();
      klass = k->predict();
      cv::putText(img3, std::to_string(klass), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(118, 185, 0),
                  1);
      std::cout << "Predicted class: " << klass;
    }
  if(event == cv::EVENT_RBUTTONDOWN) { img = cv::Mat::zeros(img.size(), img.type()); }
  if(event == cv::EVENT_MOUSEMOVE && is_drawing == 1)
    {
      cv::circle(img, cv::Point(x, y), 1, cv::Scalar(255), 30);
    }
}

int main(int argc, char *argv[])
{
  const auto mainWindow  = "Main Window";
  const auto otherWindow = "other Window";
  cv::namedWindow(mainWindow, cv::WINDOW_KEEPRATIO | cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
  cv::namedWindow(otherWindow);
  cv::setMouseCallback(mainWindow, drawCircle);
  img2 = cv::Mat::zeros(cv::Size(28, 28), CV_8UC1);
  img  = cv::Mat::zeros(cv::Size(280, 280), CV_8UC1); // cv::imread(RES_DIR "/test-4.png", 0);
  m->read_feature_vector(RES_DIR "/train-images.idx3-ubyte");
  m->read_feature_labels(RES_DIR "/train-labels.idx1-ubyte");
  m->split_data();
  m->count_classes();
  while(cv::waitKey(20) != 27) // wait until ESC is pressed
    {
      imshow("Main Window", img);
      imshow("other Window", img3);
    }
  cv::waitKey();
  cv::destroyAllWindows();
}