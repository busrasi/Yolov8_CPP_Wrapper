#include "Yolov8.h"
#include <iostream>
#include <vector>
#include <random>



YOLO_V8::YOLO_V8(MODEL_LIBS mLib, MEMORY_TYPE memType, std::string &mPath) {
	modelPath = mPath;
	modelType= mLib;
	memoryType =memType;

    //load model
    loadModel();
}

void YOLO_V8::loadModel() {
	if (modelType == CPP) {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (memoryType == GPU)
        {
            std::cout << "\nRunning on CUDA" << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        else if (memoryType == CPU)
        {
            std::cout << "\nRunning on CPU" << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
	}
    else if (modelType == LIBTORCH) {

        yolo_model = torch::jit::load(modelPath);
        yolo_model.eval();

        // Check if memoryType is GPU and CUDA is available, or if memoryType is CPU
        if ((memoryType == GPU && device.type() == torch::kCUDA) || memoryType == CPU) {
            std::cerr << "device type : " << device.type() << "\n";
            // Move model to the selected device
            yolo_model.to(device, torch::kFloat32);
        }
        else {
            std::cerr << "Requested device type does not match available resources or is not supported.\n";
        }
    }
}

cv::Mat YOLO_V8::formatToSquare(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

float YOLO_V8::letterbox(cv::Mat& input_image, cv::Mat& output_image, const std::vector<float>& target_size)
{
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        }
        else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
        cv::Size(new_shape_w, new_shape_h),
        0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(114.));
    return resize_scale;
}

float YOLO_V8::generate_scale(cv::Mat& image, const std::vector<float>& target_size)
{
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}

torch::Tensor YOLO_V8::non_max_suppression(torch::Tensor& prediction, float conf_thres, float iou_thres, int max_det)
{
    auto bs = prediction.size(0);
    auto nc = prediction.size(1) - 4;
    auto nm = prediction.size(1) - nc - 4;
    auto mi = 4 + nc;
    auto xc = prediction.index({ Slice(), Slice(4, mi) }).amax(1) > conf_thres;

    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({ "...", Slice({None, 4}) }, xywh2xyxy(prediction.index({ "...", Slice(None, 4) })));

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) {
        output.push_back(torch::zeros({ 0, 6 + nm }, prediction.device()));
    }

    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({ xc[xi] });
        auto x_split = x.split({ 4, nc, nm }, 1);
        auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({ box, conf, j.toType(torch::kFloat), mask }, 1);
        x = x.index({ conf.view(-1) > conf_thres });
        int n = x.size(0);
        if (!n) { continue; }

        // NMS
        auto c = x.index({ Slice(), Slice{5, 6} }) * 7680;
        auto boxes = x.index({ Slice(), Slice(None, 4) }) + c;
        auto scores = x.index({ Slice(), 4 });
        auto i = nms(boxes, scores, iou_thres);
        i = i.index({ Slice(None, max_det) });
        output[xi] = x.index({ i });
    }

    return torch::stack(output);
}

torch::Tensor YOLO_V8::xywh2xyxy(const torch::Tensor& x)
{
    auto y = torch::empty_like(x);
    auto dw = x.index({ "...", 2 }).div(2);
    auto dh = x.index({ "...", 3 }).div(2);
    y.index_put_({ "...", 0 }, x.index({ "...", 0 }) - dw);
    y.index_put_({ "...", 1 }, x.index({ "...", 1 }) - dh);
    y.index_put_({ "...", 2 }, x.index({ "...", 0 }) + dw);
    y.index_put_({ "...", 3 }, x.index({ "...", 1 }) + dh);
    return y;
}

torch::Tensor YOLO_V8::xyxy2xywh(const torch::Tensor& x)
{
    auto y = torch::empty_like(x);
    y.index_put_({ "...", 0 }, (x.index({ "...", 0 }) + x.index({ "...", 2 })).div(2));
    y.index_put_({ "...", 1 }, (x.index({ "...", 1 }) + x.index({ "...", 3 })).div(2));
    y.index_put_({ "...", 2 }, x.index({ "...", 2 }) - x.index({ "...", 0 }));
    y.index_put_({ "...", 3 }, x.index({ "...", 3 }) - x.index({ "...", 1 }));
    return y;
}

torch::Tensor YOLO_V8::scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape)
{
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    boxes.index_put_({ "...", 0 }, boxes.index({ "...", 0 }) - pad0);
    boxes.index_put_({ "...", 2 }, boxes.index({ "...", 2 }) - pad0);
    boxes.index_put_({ "...", 1 }, boxes.index({ "...", 1 }) - pad1);
    boxes.index_put_({ "...", 3 }, boxes.index({ "...", 3 }) - pad1);
    boxes.index_put_({ "...", Slice(None, 4) }, boxes.index({ "...", Slice(None, 4) }).div(gain));
    return boxes;
}

torch::Tensor YOLO_V8::nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold)
{
    if (bboxes.numel() == 0)
        return torch::empty({ 0 }, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kLong));

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);

            auto w = std::max(static_cast<float>(0), xx2 - xx1);
            auto h = std::max(static_cast<float>(0), yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold)
                suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}

std::vector<Detection> YOLO_V8::inference(const cv::Mat& frame)
{
    if (modelType == CPP) {
       // cv::Mat frame = cv::imread(imgPath);
        cv::Mat modelInput = frame;
        if (memoryType == GPU || memoryType == CPU)
        {
            if (letterBoxForSquare && modelShape.width == modelShape.height)
                modelInput = formatToSquare(modelInput);

            cv::Mat blob;
            cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
            net.setInput(blob);

            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());


            int rows = outputs[0].size[1];
            int dimensions = outputs[0].size[2];

           
            rows = outputs[0].size[2];
            dimensions = outputs[0].size[1];

            outputs[0] = outputs[0].reshape(1, dimensions);
            cv::transpose(outputs[0], outputs[0]);

            float* data = (float*)outputs[0].data;

            float x_factor = modelInput.cols / modelShape.width;
            float y_factor = modelInput.rows / modelShape.height;
            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;

            for (int i = 0; i < rows; ++i)
            {
                float* classes_scores = data + 4;

                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double maxClassScore;

                minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

                if (maxClassScore > modelScoreThreshold)
                {
                    confidences.push_back(maxClassScore);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
                data += dimensions;
            }
            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
            std::vector<Detection> detections{};
            for (unsigned long i = 0; i < nms_result.size(); ++i)
            {
                int idx = nms_result[i];

                Detection result;
                result.class_id = class_ids[idx];
                result.confidence = confidences[idx];

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dis(100, 255);
                result.color = cv::Scalar(dis(gen),
                    dis(gen),
                    dis(gen));
                result.className = classes[result.class_id];
                result.box = boxes[idx];

                detections.push_back(result);
            }

            return detections;
        }
    }
    else if (modelType == LIBTORCH) {
        cv::Mat input_image;
        cv::Mat image = frame;
        letterbox(image, input_image, { 1280, 1280 });
        torch::Tensor image_tensor = torch::from_blob(input_image.data, { input_image.rows, input_image.cols, 3 }, torch::kByte).to(device);
        image_tensor = image_tensor.toType(torch::kFloat32).div(255);
        image_tensor = image_tensor.permute({ 2, 0, 1 });
        image_tensor = image_tensor.unsqueeze(0);
        std::vector<torch::jit::IValue> inputs{ image_tensor };
        torch::Tensor output = yolo_model.forward(inputs).toTensor().cpu();
        // NMS
        auto keep = non_max_suppression(output)[0];
        auto boxes = keep.index({ Slice(), Slice(None, 4) });
        keep.index_put_({ Slice(), Slice(None, 4) }, scale_boxes({ input_image.rows, input_image.cols }, boxes, { image.rows, image.cols }));
        // Show the results
        std::vector<Detection> detections{};
        
        for (int i = 0; i < keep.size(0); i++) {
            Detection result;


            int x1 = keep[i][0].item().toFloat();
            int y1 = keep[i][1].item().toFloat();
            int x2 = keep[i][2].item().toFloat();
            int y2 = keep[i][3].item().toFloat();
            float conf = keep[i][4].item().toFloat();
            int cls = keep[i][5].item().toInt();
            std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << classes[cls] << std::endl;

            int x = x1;
            int y = y1;
            int w = x2 - x1;
            int h = y2 - y1;
            cv::Rect box(x,y,w,h);
            result.class_id= cls;
            result.confidence = conf;
            result.className = classes[cls];
            result.box = box;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = cv::Scalar(dis(gen),
                dis(gen),
                dis(gen));

            detections.push_back(result);
        }
        return detections;
    }
}

void YOLO_V8::inferenceV2(std::string imgPath)
{
    cv::Mat frame = cv::imread(imgPath);

    std::vector<Detection> output = inference(frame);
    int detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;

    for (int i = 0; i < detections; ++i)
    {
        Detection detection = output[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;
        // Reduce the size of the detection box and make the line thinner
        cv::rectangle(frame, box, color, 1); // Thickness reduced to 1

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        double fontScale = 0.5; // Font scale reduced to half
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, fontScale, 1, 0); // Line thickness reduced
        cv::Rect textBox(box.x, box.y - textSize.height - 10, textSize.width, textSize.height + 5);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        // Recalculate text position for center alignment
        cv::Point textOrg(box.x + (textBox.width - textSize.width) / 2, box.y - 5);
        cv::putText(frame, classString, textOrg, cv::FONT_HERSHEY_DUPLEX, fontScale, cv::Scalar(0, 0, 0), 1); // Thickness of text set to 1
    }
    // Inference ends here...

    // This is only for preview purposes
    float scale = 0.8;
    cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
    cv::imshow("Inference", frame);
    cv::waitKey(-1);
}

YOLO_V8::~YOLO_V8()
{
}

