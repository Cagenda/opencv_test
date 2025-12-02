#ifndef _YOLOV5S_H
#define _YOLOV5S_H
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <opencv2/imgproc.hpp>
// 1. 包含OpenCV核心功能头文件，定义了  cv::Mat
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <iostream>

#include "im2d.h"
#include "rga.h"

#include "rknn_api.h"
class Yolov5s
{
public:
    Yolov5s(const char *model_path, int npu_index); // 在这里为什么需要npu_index?
    ~Yolov5s();

    rknn_context ctx;                         // 模型“句柄”,之后所有 RKNN 的操作都要用这个 ctx：
    rknn_sdk_version version;                 // SDK Version
    rknn_input_output_num io_num;             // io_num
    std::vector<rknn_tensor_attr> input_attr; // rknn_tensor_attr是tens的属性
    std::vector<rknn_tensor_attr> output_attr;

    // 定义模型期望的输入图像的（高 / 宽 / 通道数）
    int model_height;
    int model_weidth;
    int model_channel;

    // 定义我们读取的视频帧的长度、宽度、和通道
    int img_height;
    int img_weidth;
    int img_channel;

    //==========记得补充注释
    int inference_image(const cv::Mat &orign_img);

    unsigned char *load_data(FILE *fp, size_t offset, size_t sz);

private:
    unsigned char *load_model(const char *file_name, int *model_size); // 把模型文件（.rknn）从文件系统读到内存，并返回这块内存的指针。
    unsigned char *model_data;                                         // 用来接收load_model的返回值（指针）
    int model_data_size;                                               // 记录 .rknn 模型文件的大小（字节数）。
};

#endif
