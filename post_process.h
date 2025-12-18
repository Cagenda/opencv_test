#ifndef _POST_PROCESS_H_
#define _POST_PROCESS_H_
#include <string>
#include <iostream>
#include <vector>
// 1. 包含OpenCV核心功能头文件，定义了  cv::Mat
#include <opencv2/highgui.hpp>
#include <fstream>
#include <algorithm>
#include "rknn_api.h"
#include <algorithm>
#include <cmath>
#define LABEL_PATH "/home/orangepi/opencv_test/model/coco_80_labels_list.txt"
#define OBJ_CLASS_NUM 80


std::vector<std::string> labels_vector; // 这一行是“真正的定义”，只在这里写一次
// ==================================定义检测结果结构体===============

// ---------------- 检测结果结构体 ,也就是检测框----------------
struct Detection
{
    int class_id;
    float score;
    float x1, y1, x2, y2; // 在原图坐标系里的左上角(x1,y1) 右下角(x2,y2)
};
int post_process();
// ---------------- 真正的 YOLOv5 后处理入口 ----------------
// 注意：这里假定 outputs 里的 buf 现在是 int8_t*（因为 want_float = 0）
//       函数内部会根据 out_attrs[i].scale / zp 做反量化
std::vector<Detection> yolov5_post_process(
    const rknn_output *outputs,        // rknn_outputs_get 得到的输出数组
    const std::vector<rknn_tensor_attr> &  out_attrs,  // ✅ 改成引用 vector 对应的 tensor 属性数组（包含 scale / zp / dims）
    int out_num,                       // 输出个数=3
    int model_w,                       // 640
    int model_h,                       // 640
    int img_w,                         // 原图宽
    int img_h,                         // 原图高
    float conf_thres,                  // 例如 0.25
    float nms_thres                    // 例如 0.45
);

#endif