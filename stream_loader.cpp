#include "stream_loader.h"

// 先进行判断【工具函数】判断数据包是否为 H.264 的 Annex-B 格式 (00 00 00 01)
//  Annex B 格式以 0x000001 或 0x00000001 开头
int is_annexb(const uint8_t *buf, size_t buf_size)
{ // 需要你弄懂，buf从何而来？？？
    if (buf_size >= 4)
    {
        if (buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x01 ||
            (buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x00 && buf[3] == 0x01))
        {
            return 1;
        }
    }
    else
    {
        return 0;
    }
    return 0;
}

StreamLoader::StreamLoader(std::string url, int id)
{
    // =先把ID和目标(URL)记在心里
    this->stream_loader_id = id;
    this->stream_url = url;

    // 这两个参数含义？
    this->status = 0;
    this->count = 0;
    // [逻辑] 设置窗口名字，方便 OpenCV 显示时区分是谁
    this->windowsName = std::to_string(id);

    std::cout << "[StreamLoader] Created loader for ID: " << id << std::endl;
}

StreamLoader::~StreamLoader()
{
    // [逻辑] 临死前，必须把手里的资源（内存、网络连接）都交出去
    this->close(); // 暂时没写
    std::cout << "[StreamLoader] Destroyed loader ID: " << stream_loader_id << std::endl;
}

void StreamLoader::open()

{
    int ret;
    // AVDictionary是一个 FFmpeg 用来传“打开输入流时的参数/配置”的字典（键值对表），你用 av_dict_set() 往里面塞参数，然后把它传给 avformat_open_input()，让 FFmpeg 按这些参数去连接 RTSP
    AVDictionary *options = NULL;
    //==============1.配置FFmpeg参数==================
    // [逻辑] 必须用 TCP，否则 UDP 丢包会导致花屏
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    // [逻辑] 设置缓冲区大小，防止高码率视频卡顿
    av_dict_set(&options, "buffer_size", "8192000", 0);
    // [逻辑] 设置最大延迟 0.5秒，保证实时性
    av_dict_set(&options, "max_delay", "500000", 0);
    // [逻辑] 设置超时 5秒。如果网线被拔了，不能一直卡死在这里，要能返回错误
    av_dict_set(&options, "stimeout", "5000000", 0);

    // ===============2.打开RTSP流====================
    // fmtCtx意义：FFmpeg 打开 RTSP 流后，把所有“这个流的全局信息 + 读取状态”都放进 fmtCtx 里
    ret = avformat_open_input(&fmtCtx, stream_url.c_str(), 0, &options);

    if (ret < 0)
    {
        std::cerr << "[Error] Failed to open rtsp: " << stream_url << std::endl;
        status = 1; // 标记失败，operator() 会看到这个标志并尝试重连
        return;
    }

    // ============= 3. 解析流信息 ===================
    // 因为很多流（尤其 RTSP）在 avformat_open_input() 时拿到的信息不完整。avformat_find_stream_info() 会继续从网络里读一段数据，分析包头、时间戳、关键帧等，来把 fmtCtx->streams[i] 的字段补齐。
    ret = avformat_find_stream_info(fmtCtx, 0);
    if (ret < 0)
    {
        std::cerr << "[Error] Failed to find stream info" << std::endl;
        status = 1; // 这里状态置为1是什么意思？
        return;
    }
    // ============== 4. 寻找视频流 ==================
    // 一个 RTSP 可能包含视频、音频、字幕。我们要找到“视频”那一轨。
    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++)
    {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoStreamIndex = i;
            break;
        }
    }
    if (videoStreamIndex == -1) // 没找到视频流
    {
        std::cerr << "[Error] No video stream found" << std::endl;
        status = 1;
        return;
    }

    //============5.准备过滤器（AVCC->Annex-b）=========
    // MPP 硬件解码器只吃 Annex-B (00 00 00 01) 格式。
    // 大部分网络流是 AVCC 格式，所以需要这个过滤器来转换
    bsf = av_bsf_get_by_name("h264_mp4toannexb");
    if (!bsf)
    {
        std::cerr << "[Error] Failed to find h264_mp4toannexb filter" << std::endl;
        status = 1;
        return;
    }

    av_bsf_alloc(bsf, &bsf_ctx); // 根据过滤器类型 bsf 创建一个可使用的过滤器实例 bsf_ctx
    // 复制原流的参数给过滤器
    avcodec_parameters_copy(bsf_ctx->par_in, fmtCtx->streams[videoStreamIndex]->codecpar);
    av_bsf_init(bsf_ctx);
    //===============6.初始化MPP解码器==================
    decoder.Init(264, 25, this, stream_loader_id); // 这里的this是什么意思？？？this = 当前 StreamLoader 对象的指针

    // ================= 【新增关键修复】 =================
    // 很多 RTSP 流的 SPS/PPS 信息藏在 extradata 里，必须先喂给 MPP，
    // 否则 MPP 收到第一个 I 帧时会因为缺少参数而丢弃！
    if (fmtCtx->streams[videoStreamIndex]->codecpar->extradata_size > 0)
    {
        uint8_t *extra_data = fmtCtx->streams[videoStreamIndex]->codecpar->extradata;
        int extra_size = fmtCtx->streams[videoStreamIndex]->codecpar->extradata_size;

        std::cout << "[StreamLoader] Feeding extradata (SPS/PPS) to MPP, size: " << extra_size << std::endl;

        // 发送 SPS/PPS 数据包
        decoder.Decode(extra_data, extra_size, 0);
    }
    // ===================================================
    decoder.SetCallback(mpp_decoder_frame_callback); // 回调函数暂时没写
    // ============= 6. 分配接收包内存 =================
    // 向 FFmpeg 申请（分配）一个 AVPacket 结构体，用来装“从网络/文件读取出来的一包压缩码流数据

    temp_pkt = av_packet_alloc(); // 估计是在read中用到

    status = 0; // 一切顺利，标记正常
    std::cout << "[Success] Stream opened: " << stream_url << std::endl;
}

// 【核心回调】MPP 解码完成后的处理逻辑
// 这个函数是全局的（或静态的），它不属于任何对象，所以它没有 'this' 指针。但是！MPP 会把我们在 Init 时传入的 'this' 指针，通过 'userdata' 参数还给我们。
void mpp_decoder_frame_callback(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data, int id)
{
    StreamLoader *self = (StreamLoader *)userdata;
    if (data == NULL)
    {
        std::cout << "callback" << std::endl;
        return;
    }
    //=========0.申请没有垃圾填充的内存================
    cv::Mat yuvMat(height + height / 2, width, CV_8UC1);
    // ==========1.获取源数据 (硬件内存) 的起始地址=====
    uint8_t *src_base = (uint8_t *)data;
    //====2.获取目标数据 (我们申请的干净内存) 的起始地址==
    uint8_t *dst_ptr = yuvMat.data;
    // --- A. 搬运 Y 分量 (亮度) ---
    // 此时 src_base 指向 Y 平面开头
    for (int i = 0; i < height; i++)
    {
        // 只拷贝有效数据 (width)，跳过末尾的填充数据
        memcpy(dst_ptr, src_base, width);
        src_base = src_base + width_stride;
        dst_ptr = dst_ptr + width;
    }

    // --- B. 搬运 UV 分量 (色度) ---
    // 计算 UV 数据在硬件内存中的起始位置
    // 必须跳过完整的 height_stride 行，且每行长度是 width_stride
    uint8_t *src_uv = (uint8_t *)data + width_stride * height_stride;
    for (size_t i = 0; i < height / 2; i++)
    {
        memcpy(dst_ptr, src_uv, width);
        dst_ptr = dst_ptr + width;
        src_uv = src_uv + width_stride;
    }

    //=====================格式转换=====================
    // 此时 yuvMat 已经是整理好的、连续的内存了，OpenCV 可以完美处理
    // 结果存入 self->bgrMat
    cv::cvtColor(yuvMat, self->bgrMat, cv::COLOR_YUV2BGR_NV12);

    // 4. 【计数更新】
    self->count++;
}

// 从 FFmpeg 拿数据 -> 判断格式 -> 喂给 MPP
// int StreamLoader::read_frame()
// {
//     // 1. 从网络读取一帧数据包 (Packet)
//     // 结果存放在 this->temp_pkt 中
//     int ret = av_read_frame(fmtCtx, temp_pkt);
//     if (ret < 0)
//     {
//         // 读取失败（可能是断网、服务器断开或文件结束）
//         // 返回 -1 通知调用者（operator()）进行重连处理
//         std::cout << "av_read_frame 读取失败" << std::endl;
//         av_packet_unref(temp_pkt);
//         return -1;
//     }
//     // 检查数据包里是不是视频流,现在为止数据包里的数据流依旧是音频视频等等结合在一块
//     if (temp_pkt->stream_index == videoStreamIndex)
//     {
//         // 确认是视频,进行格式判断是不是annexb

//         if (!is_annexb(temp_pkt->data, temp_pkt->size))
//         {
//             ret = av_bsf_send_packet(bsf_ctx, temp_pkt);
//             if (ret < 0)
//             {
//                 av_packet_unref(temp_pkt); // 发送失败也要释放内存
//                 std::cout << "av_send_packet 失败" << std::endl;
//                 return 0; // 跳过这帧
//             }

//             // 从过滤器中转好好的包。一个输入包可能会被过滤器拆成多个输出包，所以用 while 循环。
//             while (av_bsf_receive_packet(bsf_ctx, temp_pkt) == 0)
//             {
//                 // 此时 temp_pkt->data 已经是带有 00000001 的标准 H.264 了
//                 // 喂给 MPP 解码器！
//                 ret = decoder.Decode(temp_pkt->data, temp_pkt->size, 0);
//                 if (ret < 0)
//                 {
//                     std::cout << "解码失败" << std::endl;
//                     av_packet_unref(temp_pkt);
//                     return -1;
//                 }
//                 av_packet_unref(temp_pkt);

//             }
//         }
//         else
//         {
//             // 不需要数据转换
//             ret = decoder.Decode(temp_pkt->data, temp_pkt->size, 0);
//             if (ret < 0)
//             {
//                 std::cout << "解码失败" << std::endl;
//                 av_packet_unref(temp_pkt);
//                 return -1;
//             }
//         }
//     }
//     // 4. 【必须】释放包的引用计数
//     // 这里的 temp_pkt 只是一个容器，用完了要把内存还给 FFmpeg 的内存池
//     // 如果不写这行，内存会瞬间爆炸！
//     av_packet_unref(temp_pkt);
//     return 0; // 返回 0 表示读取正常
// }
// [stream_loader.cpp] -> read_frame() 完全替换

int StreamLoader::read_frame()
{
    // 1. 从网络读取一帧,结果存放在 this->temp_pkt 中
    int ret = av_read_frame(fmtCtx, temp_pkt);
    if (ret < 0)
    {
        // 这一步不需要打印太多错误，否则断流时会刷屏
        av_packet_unref(temp_pkt);
        return -1;
    }

    // 检查数据包里是不是视频流,现在为止数据包里的数据流依旧是音频视频等等结合在一块。之前在open函数中找到的是videoStreamIndex这个值
    if (temp_pkt->stream_index == videoStreamIndex)
    {
        // 策略：为了稳健，建议全部通过过滤器处理 SPS/PPS 的自动插入
        // 但根据经验，始终通过过滤器是最安全的，因为它会处理边界情况。

        // 发送包给过滤器
        ret = av_bsf_send_packet(bsf_ctx, temp_pkt);
        if (ret < 0)
        {
            // 发送失败（极少见），释放原包
            av_packet_unref(temp_pkt);
            return 0;
        }

        // 在这里为什么要while循环读取？因为 1 个输入 packet 可能产生多个输出 packet
        /*所以必须循环 receive 直到：

EAGAIN：当前没有更多输出了，需要再 send 新输入

EOF：filter flush 完毕（通常在结束时出现）

<0：其他错误*/
        while (true)
        {
            // 接收转换后的包（可能 1 个变 N 个）
            ret = av_bsf_receive_packet(bsf_ctx, temp_pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            {
                break; // 读完了
            }
            if (ret < 0)
            {
                break; // 出错
            }

            // 喂给 MPP
            decoder.Decode(temp_pkt->data, temp_pkt->size, 0);

            // 【关键修复】循环内必须释放，否则下一次 receive 会内存泄漏或数据错误！
            av_packet_unref(temp_pkt);
        }
    }
    else
    {
        // 非视频包，释放
        av_packet_unref(temp_pkt);
    }

    return 0;
}
// close函数
void StreamLoader::close()
{
    // 1. 重置 MPP 解码器
    // 这会释放 MPP 内部申请的硬件缓冲区，防止显存泄露
    decoder.Reset();

    // 2. 释放 FFmpeg 过滤器上下文 (h264_mp4toannexb)
    // 必须判断指针是否为空，防止重复释放导致崩溃
    if (bsf_ctx)
    {
        av_bsf_free(&bsf_ctx);
        bsf_ctx = nullptr; // 【关键】释放后必须置空，防止变成野指针
    }

    // 3. 关闭网络流连接 & 释放上下文
    // avformat_close_input 这个函数做了两件事：
    // a. 断开 TCP/RTSP 网络连接
    // b. 释放 fmtCtx 结构体内存，并将 fmtCtx 指针置为 NULL
    if (fmtCtx)
    {
        avformat_close_input(&fmtCtx);
        fmtCtx = nullptr; // 虽然函数内部会置空，但手动写一下更保险
    }
    // 4. 释放临时数据包内存
    if (temp_pkt)
    {
        av_packet_free(&temp_pkt);
        temp_pkt = nullptr;
    }
    // 打印日志方便调试，确认资源确实被关了
    std::cout << "[StreamLoader] Closed resources for ID: " << stream_loader_id << std::endl;
}

//===================线程入口========================
void StreamLoader::operator()()
{
    // 1. 线程启动时的冷启动
    // 一旦 std::thread 开始运行，就会执行这里
    this->open();

    while (1)
    {
        // ============================================================
        // A. 断线重连机制 (Self-healing)
        // ============================================================
        // 如果 status != 0，说明之前 open 失败了，或者 read_frame 报错了
        if (this->status)
        {
            // std::cout << "[StreamLoader] Connection lost, reconnecting..." << std::endl;

            this->close(); // 【关键】先清理旧的脏数据
            this->open();  // 重新尝试建立连接

            // 如果重连依然失败，不要死循环狂刷，休息 1 秒再试
            // 这样既能自动恢复，又不会把 CPU 跑满
            if (this->status)
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue; // 跳过本次循环，继续重试
            }
        }

        // ============================================================
        // B. 主循环读取 (Main Loop)
        // ============================================================
        // 尝试读取一帧
        // 注意：read_frame 内部调用的 av_read_frame 通常是阻塞的
        // 所以这里不会疯狂空转，它会等待网络包到来
        if (this->read_frame() < 0)
        {
            // 如果读取返回 -1 (比如流结束了，或者网络断了)
            // 标记 status = 1，下一次循环就会自动进入上面的重连逻辑
            std::cerr << "[StreamLoader] Read error." << std::endl;
            this->status = 1;
        }

        // (可选) 极其微小的休眠，防止在某些极端非阻塞情况下 CPU 100%
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}
