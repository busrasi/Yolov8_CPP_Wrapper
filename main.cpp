#include "Yolov8.h"

// Model Paths
std::string modelPath = "C:/Users/RevTec/source/repos/Yolo8_Wrapper_class/Yolo8_Wrapper_class/best.onnx";
std::string modelPathPt = "C:/Users/RevTec/source/repos/Yolo8_Wrapper_class/Yolo8_Wrapper_class/best.torchscript";
std::string imagePath= "C:/Users/RevTec/source/repos/Yolo8_Wrapper_class/Yolo8_Wrapper_class/test2.jpg" ;
int main(int argc, char** argv)
{
	// Libtorch
	// YOLO_V8 yolo_model(LIBTORCH,CPU, modelPathPt); 
	// yolo_model.inferenceV2(imagePath);

	//CPP 
	 YOLO_V8 yolo_model(CPP, GPU, modelPath);		
	 cv::Mat frame = cv::imread(imagePath);
	 yolo_model.inference(frame);
	 yolo_model.inferenceV2(imagePath);
	



}