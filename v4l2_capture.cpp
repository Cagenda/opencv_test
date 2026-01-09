#include "v4l2_capture.h"

// 构造函数初始化,初始化变量与数组
V4L2Capture::V4L2Capture(const char *dev_name, int width, int height)
    : dev_name_(dev_name), width_(width), height_(height), fd(-1), buffer_count_(4)
{
}

// 析构函数
V4L2Capture::~V4L2Capture()
{
    close_device();
}

int V4L2Capture::open_device()
{
    int ret = 0;
    //============================================1. 打开设备 ========================================
    fd = open(dev_name_.c_str(), O_RDWR); // fd是设备句柄，要和fp(文件句柄区分)
    if (fd == -1)
    {
        perror("open video device");
        return -1;
    }
    //========================================2. 获取设备能力 ========================================
    struct v4l2_capability cap;
    ret = ioctl(fd, VIDIOC_QUERYCAP, &cap);
    if (ret == -1)
    {
        perror("get video cap");
        return -1;
    }

    // 检查是否支持视频捕获
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
    {
        fprintf(stderr, "Device doesn't supports video capture.\n");
        return -1;
    }
    else
    {
        printf("Device supports video capture.\n");
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING))
    {
        fprintf(stderr, "Device doesn't support streaming I/O (REQBUFS/MMAP).\n");
        return -1;
    }
    else
    {
        printf("Device supports streaming.\n");
    }

    //===================================3. 设置视频格式 ==============================================

    struct v4l2_format fmt_video; // 带标签的联合体
    // type 是 标签(tag)：告诉内核“我这次要设置/查询的是哪种 buffer queue”
    // fmt 是 union：里面放了多种格式结构（pix, pix_mp, …），但同一时间只有一种是“被解释的那一种”
    memset(&fmt_video, 0, sizeof(fmt_video)); // 0. 必须清零

    // 3.1==========设置type============
    fmt_video.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 同一个设备节点（fd）同时拥有输入队列和输出队列。

    // 3.2 ======具体的参数 (根据你的 USB 摄像头实际能力)=======
    fmt_video.fmt.pix.width = width_;   // 想要的分辨率宽
    fmt_video.fmt.pix.height = height_; // 想要的分辨率高
    // 【重点】USB 摄像头为了保帧率，通常必须选 MJPEG (压缩格式)
    //  如果选 YUYV (无压缩)，带宽不够可能只能跑 5fps
    fmt_video.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    // 扫描方式：现在的摄像头全是逐行扫描，填 NONE 即可
    fmt_video.fmt.pix.field = V4L2_FIELD_NONE;
    // 3.3 =========设置=========
    ret = ioctl(fd, VIDIOC_S_FMT, &fmt_video);
    if (ret == -1)
    {
        perror("set video format ");
        return -1;
    }
    // 【新增】反向更新：以驱动实际给出的参数为准。在调用 VIDIOC_S_FMT 时，你传入了想要的 1280x720。但是，某些摄像头如果不支持这个特定分辨率，驱动程序可能会“自作主张”地把你修改为它支持的最接近的分辨率（比如 1024x768 或 640x480）。
    if (width_ != fmt_video.fmt.pix.width || height_ != fmt_video.fmt.pix.height)
    {
        printf("Warning: Driver changed resolution from %dx%d to %dx%d\n",
               width_, height_, fmt_video.fmt.pix.width, fmt_video.fmt.pix.height);
        width_ = fmt_video.fmt.pix.width;
        height_ = fmt_video.fmt.pix.height;
    }

    //======================================4. 向内核申请内存缓冲区=================================
    //============这一步的命令是 VIDIOC_REQBUFS。
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req)); // 清0
    req.count = buffer_count_;    /// 1.==============最少 2 (双缓冲)，但 4 个更稳，能防止应用层处理太慢导致丢帧。
    // 2.================type: 必须和之前的 S_FMT 保持一致！(敲黑板)你之前填的是 V4L2_BUF_TYPE_VIDEO_CAPTURE，这里必须一模一样。申请的这些内存，是专门给视频采集队列用的
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    // 3.================memory: 内存映射方式 USB 摄像头最通用的是 MMAP (Memory Map)。意思是：内核在内核空间申请内存，然后映射到用户空间给你用。
    req.memory = V4L2_MEMORY_MMAP;
    ret = ioctl(fd, VIDIOC_REQBUFS, &req); // 将req交给内核
    if (ret == -1)
    {
        perror("request buffer");
        return -1;
    }
    //=======================================5. 内存映射============================================

    bufs_vector.resize(req.count);
    // 驱动可能不给你 4 个，只给你更少.它会把结果写回 req.count
    if (req.count < 2)
    {
        fprintf(stderr, "Insufficient buffer memory: got %u\n", req.count);
        return -1;
    }
    for (int i = 0; i < req.count; i++)
    {
        struct v4l2_buffer buf_v;         // 此时buf.length 是不确定/没意义的。同时需要注意buf.length代表的是 容量
        memset(&buf_v, 0, sizeof(buf_v)); // 声明并清零 v4l2_buffer，每次循环都需要清零，每一个buf的i是不同的
        buf_v.type = req.type;            // 和设置的fmt格式一样VIDEO_CAPTURE
        buf_v.memory = V4L2_MEMORY_MMAP;  // 告诉内核：我使用的是 mmap 方式申请的 buffer  和 REQBUFS 里的 req.memory 必须一致
        buf_v.index = i;                  // 告诉内核：我要查询第几个 buffer（0..count-1）
        ret = ioctl(fd, VIDIOC_QUERYBUF, &buf_v);
        /*你的程序陷入内核态。驱动程序根据你传入的 index，去内部表格里查到了第 i 号缓冲区的物理信息。结果： ioctl 返回成功后，buf 结构体被内核修改了！
    buf.length：变成真的长度（比如 1843200 字节）。
    buf.m.offset：变成真的物理偏移量（比如 0x100000*/
        if (ret == -1)
        {
            perror("VIDIOC_QUERYBUF");
            return -1;
        }
        bufs_vector[i].length = buf_v.length; // 记录信息，查询和操作视频设备缓冲区信息的工具
        bufs_vector[i].start = mmap(NULL, buf_v.length,
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED,
                                    fd, buf_v.m.offset); // 把第 i 块内核 buffer 映射到用户态，保存到...
        if (bufs_vector[i].start == MAP_FAILED)
        {
            perror("mmap");
            return -1;
        }
        else
        {
            std::cout << bufs_vector[i].start << std::endl;
        }
    }

    // =========================================6. 把所有缓冲区入队 ==================================
    // 阶段 1】启动前：一次性把 4 个盘子全扔进去
    for (__u32 i = 0; i < req.count; i++)
    {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf)); // 1. 清零是必须的

        // 2. 填写入队申请单
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 赛道：视频采集
        buf.memory = V4L2_MEMORY_MMAP;          // 方式：内存映射
        buf.index = i;                          // 核心：告诉内核“我把第 i 号盘子还给你”

        // 3. 执行入队动作
        // 这一步之后，第 i 号 buffer 的“所有权”就从【用户】转移到了【内核】
        ret = ioctl(fd, VIDIOC_QBUF, &buf); // QUBUF将缓冲区放入队列。从这一步开始，驱动获得该 buffer 的使用权
        if (ret == -1)
        {
            perror("VIDIOC_QBUF");
            return -1;
        }
    }
    // 此时：你手里 0 个盘子，内核手里 4 个盘子。
    printf("All buffers queued successfully.\n");
    // =================================== 7. 开启视频流 STREAMON =================================
    // 执行成功后，摄像头硬件开始工作，自动把图像数据填入你刚才 QBUF 的缓冲区里
    v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(fd, VIDIOC_STREAMON, &type);
    if (ret == -1)
    {
        perror("VIDIOC_STREAMON");
        return -1;
    }

    return 0;
}
// ================================8.获取一帧数据 (Polling + DQBUF)==================================
int V4L2Capture::get_frame(void *&addr, int &size, int &index)
{

    // =================================== 8. 采集循环  ===========================
    // 取出一个盘子就需要还一个盘子
    // VIDIOC_DQBUF从驱动的“已完成队列(done queue)”里，把一个 buffer 的“所有权”交还给用户态，并通过 struct v4l2_buffer 告诉你是哪一个 index、这一帧用了多少字节等元数据。
    struct pollfd fds[1];   // 定义一个poll数组，因为 poll 可以同时让保安盯好几个设备
    fds[0].fd = fd;         // 【监视对象】：摄像头的文件描述符
    fds[0].events = POLLIN; // 【监视事件】：POLLIN 代表 "有数据可读" (Poll Input)

    // 3. 开启监视 (进入休眠)
    int ret = poll(fds, 1, 2000);
    // --- 程序运行到这里，会暂停（Blocked），CPU 切换去做别的事 ---
    // --- 直到：1. 有数据来了； 或者 2. 2秒钟到了 ---
    // 4. 判断结果
    if (ret < 0)
    {
        perror("poll error"); // 系统错误（如被信号中断）
        return -1;
    }
    else if (ret == 0)
    {
        // 此时是被“超时”叫醒的
        printf("2秒到了,还是是没数据，可能摄像头坏了！\n");
        return -1;
    }
    else
    {
        // --- 正常处理 ---
        if (fds[0].revents & POLLIN) // 确认是 POLLIN 事件（数据到了）
        {
            // A. 准备结构体 (Empty Container)
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 指定流类型
            buf.memory = V4L2_MEMORY_MMAP;          // 指定内存模式

            // B. 取出填充好的 Buffer (DQBUF - Dequeue)
            // 内核把“使用权”交给你，并填入 buf.index 和 buf.bytesused。在这里因为将第几个盘子的index返回，以及返回
            if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1)
            {
                perror("VIDIOC_DQBUF");
                return -1;
            }

            // C. 将关键信息传出给调用者
            index = buf.index;
            size = buf.bytesused;
            addr = bufs_vector[buf.index].start; // 找到对应的内存首地址
            return 0;
        }
    }
    return -1;
}
//====================================9.归还一帧数据帧===============================================
int V4L2Capture::put_frame(const int index) // 在这里为什么我不用int & index。引用的底层实现其实是指针。在 64 位系统上，指针占 8 个字节。int 通常只有 4 个字节。
{
    // QBUF (把盘子还给内核，让它继续去接新的水)
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index; // 必须填入刚才取出的索引

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1)
    {
        perror("VIDIOC_QBUF");
        return -1;
    }
    return 0;
}

//========================================10.关闭并停止=====================================================
void V4L2Capture::close_device()
{
    // 10.1关流
    v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1)
    {
        perror("VIDIOC_STREAMOFF");
    }
    // 10.2解除内存映射
    for (size_t i = 0; i < bufs_vector.size(); ++i)
    {
        if (bufs_vector[i].start != MAP_FAILED && bufs_vector[i].start != NULL)
        {
            munmap(bufs_vector[i].start, bufs_vector[i].length);
        }
    }
    bufs_vector.clear(); // 清空 vector
    // 10. 3. 关闭文件描述符
    close(fd);
    printf("V4L2: Device Closed.\n");
}
