// 你要把 demo_multhread.cpp 里的逻辑，封装成一个叫 StreamLoader 的新零件。 这个零件包含两个核心能力：拉流 (FFmpeg)：负责把 RTSP 网络数据包拉下来。解码 (MPP)： 负责把 H.264 压缩包利用硬件解压成图像
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>

// 2. 引入 OpenCV
#include <opencv2/opencv.hpp>

// 3. 引入 FFmpeg (必须用 extern "C" 包裹)
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
}
#include "mpp_decoder.h"
void mpp_decoder_frame_callback(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data, int id); // 是一个普通的全局函数,哪个对象都可以调用
class StreamLoader
{
public:
    // 传入 RTSP 地址和 ID
    StreamLoader(std::string url, int id);

    ~StreamLoader();
    void open();
    void operator()();


    int read_frame();

    void close();

public:
    int stream_loader_id;
    std::string stream_url;
    std::string windowsName; //???
    // 这俩个参数意义？？？
    int status;
    int count;
    int videoStreamIndex = -1; // 一个RTSP 可能包含视频、音频、字幕。我们要找到“视频”那一轨，因此需要循环遍历

    cv::Mat bgrMat;
    // 存放转换后的BGR 图像（YOLO用这个)

    AVPacket *temp_pkt = nullptr; // 在read函数中调用
    AVFormatContext *fmtCtx = nullptr;
    // 过滤器相关（用于处理H.264格式)
    const AVBitStreamFilter *bsf = nullptr; // bsf指向可以变
    AVBSFContext *bsf_ctx = nullptr;

    MppDecoder decoder; // 解码器实例
};
