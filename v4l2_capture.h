#ifndef V4L2_CAPTURE_H
#define V4L2_CAPTURE_H
#include <vector>
#include <string>
#include <linux/videodev2.h> // 需要用到里面的宏定义
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <string>
#include <iostream>
// 定义一个简单的结构体来管理映射后的内存
struct buffer
{
    void *start;
    size_t length; // 这里的 length 是 buffer 的总容量
};

class V4L2Capture
{
public:
    // 构造函数：传入设备名(如 /dev/video0)，宽，高
    V4L2Capture(const char *dev_name, int width, int height);

    // 析构函数：自动关闭设备
    ~V4L2Capture();

    // 1. 打开设备并初始化 (Open -> Init -> Mmap -> StreamOn)
    // 返回 0 成功，-1 失败
    int open_device();

    // 2. 获取一帧数据 (Polling + DQBUF)
    // 参数是输出参数：
    // *addr:  将会指向图像数据的内存首地址
    // *size:  将会拿到这一帧的实际大小 (bytesused)
    // *index: 将会拿到这一帧的 buffer 索引 (用于稍后归还)
    // 返回 0 成功，-1 失败
    int get_frame(void *&addr, int &size, int &index);

    // 3. 归还一帧数据 (QBUF)
    // index: 刚才 get_frame 拿到的索引
    int put_frame(const int index);

    // 4. 停止并关闭 (StreamOff -> Munmap -> Close)
    void close_device();

private:
    std::string dev_name_;
    int width_;
    int height_;
    int fd; // 设备文件描述符

    // 缓冲区管理
    std::vector<buffer> bufs_vector; // 管理映射的内存
    int buffer_count_;               // 申请的 buffer 数量 (通常是 4)
};

#endif // V4L2_CAPTURE_H