#pragma once
#include <string>
// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <torch/torch.h>
#include <torch/script.h>


using torch::indexing::Slice;
using torch::indexing::None;

enum MODEL_LIBS
{
    CPP = 1,
    LIBTORCH = 2
};
enum MEMORY_TYPE
{
    GPU = 1,
    CPU = 2,
};

struct Detection
{
    int class_id{ 0 };
    std::string className{};
    float confidence{ 0.0 };
    cv::Scalar color{};
    cv::Rect box{};
};
class YOLO_V8
{

public:
    YOLO_V8(MODEL_LIBS,MEMORY_TYPE, std::string& onnxModelPath);

    void loadModel();
    std::vector<Detection> inference(const cv::Mat& input);
    void inferenceV2(std::string imgPath);
    ~YOLO_V8();

    cv::Mat formatToSquare(const cv::Mat& source);


public:
    
    std::string modelPath{};
    std::string imagePath{};
    MODEL_LIBS modelType{};
    MEMORY_TYPE memoryType{};

    // CLASSES
    std::vector<std::string> classes = {
    "2c_s", "2d_s", "2h_s", "2s_s",
    "3c_s", "3d_s", "3h_s", "3s_s",
    "4c_s", "4d_s", "4h_s", "4s_s",
    "5c_s", "5d_s", "5h_s", "5s_s",
    "6c_s", "6d_s", "6h_s", "6s_s",
    "7c_s", "7d_s", "7h_s", "7s_s",
    "8c_s", "8d_s", "8h_s", "8s_s",
    "9c_s", "9d_s", "9h_s", "9s_s",
    "Tc_s", "Td_s", "Th_s", "Ts_s",
    "Jc_s", "Jd_s", "Jh_s", "Js_s",
    "Qc_s", "Qd_s", "Qh_s", "Qs_s",
    "Kc_s", "Kd_s", "Kh_s", "Ks_s",
    "Ac_s", "Ad_s", "Ah_s", "As_s",
    "chips"
    };
    cv::dnn::Net net;
    
    cv::Size2f modelShape{1280,1280};

    float modelConfidenceThreshold{ 0.25 };
    float modelScoreThreshold{ 0.45 };
    float modelNMSThreshold{ 0.50 };
    
    bool letterBoxForSquare = true;

    // for LibTorch
    float letterbox(cv::Mat& input_image, cv::Mat& output_image, const std::vector<float>& target_size);
    float generate_scale(cv::Mat& image, const std::vector<float>& target_size);
    torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300);
    torch::Tensor xywh2xyxy(const torch::Tensor& x);
    torch::Tensor xyxy2xywh(const torch::Tensor& x);
    torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape);
    torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);

    torch::jit::script::Module yolo_model;
    torch::Device device = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
};