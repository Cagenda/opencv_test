#include <iostream> // 用于标准输入输出 (cout)
#include <opencv2/core.hpp>
// 1. 包含OpenCV核心功能头文件，定义了  cv::Mat
#include <opencv2/highgui.hpp>
#include <queue>
#include <thread>
#include "SafeQueue.h"
#include <mutex>
#include <map>
#include "yolov5s.h"
#include "post_process.h"
#include "thread_pool.h"
#include "v4l2_capture.h"
// 2. 包含图像I/O和GUI函数头文件， 定义了 imread, imshow, waitKey#include <iostream>
#include "stream_loader.h"
#include "gb28181_agent.h"

using namespace std;
using namespace cv;
// RK3588 有 3 个 NPU 核心，我们创建 6 个实例来保证流水线充盈
ThreadPool gthreadpool(12, "/home/orangepi/opencv_test/model/yolov5s.rknn", 3); // 定义一个线程池，这个线程池中有12个线程
static int g_frame_start_id = 0;                                                // 用于帧的起始id

// 定义每一帧（也就是每一张的图片）的信息，在这里新增了频道的id
struct FrameData
{
    cv::Mat frame;  // 创建一个Mat类型的图片（也就是一帧）
    int index;      // 帧索引
    int channel_id; // 摄像头的身份证号 (0, 1, 2, 3...)
};
// 定义“流水线任务”结构体：我们需要一个结构体来暂存“发出去的订单”。
struct PendingTask
{
    int index;                // 帧序号
    int channerl_id;          // 【新增】记住这是第几路摄像头的任务
    std::future<cv::Mat> fut; // 取餐票 (注意这里是 future<Mat>)
};

// 【新增】显示缓存：存放每一路最新的 BGR 图片，给拼图线程用
const int MAX_CHANNELS = 4;                        // 支持4路视频拉流
std::vector<cv::Mat> display_buffer(MAX_CHANNELS); // 定义一个容器，并且预先存放4个空的Mat
std::mutex display_mutex;                          // 保护上面的缓存

// 【新增】结果缓存：存放每一路最新的检测结果，给拼图线程画框用
std::vector<std::vector<Detection>> result_buffer(MAX_CHANNELS);
std::mutex result_mutex; // 保护上面的缓存

// 创建读取视频队列
SafeQueue<FrameData> SafeQueue_Read;

// 创建写入视频的队列
SafeQueue<FrameData> SafeQueue_Write;

// 1. 【生产者】线程函数：只能有一个线程执行它（有mutex）
// 它的职责是顺序读取视频，并安全地将帧放入队列
void Thread_ReadVideo(std::vector<std::shared_ptr<StreamLoader>> &cameras, SafeQueue<FrameData> &img_queue, int &img_index, mutex &cap_mutex, bool &finish)
{
    printf("Read Thread Started (Multi-Channel Polling Mode)...\n");

    // 启动 StreamLoader 的拉流线程（就在这个函数里跑死循环，或者调用 operator()）
    // 但注意：你的 StreamLoader 设计是 operator() 里面自带死循环。
    // 所以我们不能在这里调用 camera()，否则会卡死在这里，无法往下执行入队逻辑。

    // 【关键修正】：
    // 你的 StreamLoader 设计是：operator() 负责拉流+解码，把图存到 bgrMat。
    // 所以我们需要把 camera() 放在一个独立线程里跑（在 main 函数里启动）。
    // 而这个 Thread_ReadVideo 只需要负责“搬运”：把 camera.bgrMat 搬运到 img_queue。

    while (1) // 独立的搬运线程
    {
        // ========== 【现在可以用了】 =================
        // 检查仓库水位：如果积压超过 30 帧，就暂停进货
        if (img_queue.size() > 20)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue; // 主动丢弃这一帧，防止内存爆炸
        }
        //============================================
        bool has_new_data = false; // 标记这一轮循环有没有抓到图
        // ===== B. 轮询所有摄像头 ===============
        for (int i = 0; i < cameras.size(); i++)
        {
            cv::Mat frame_copy;
            bool ready = false;
            {
                std::lock_guard<std::mutex> lock(cameras[i]->mat_mutex);
                if (!cameras[i]->bgrMat.empty())
                {
                    frame_copy = cameras[i]->bgrMat.clone();
                    ready = true;
                }
            }
            if (ready)
            {
                // 全局序号自增
                int current_idx;
                {
                    std::lock_guard<mutex> lock(cap_mutex);
                    img_index++;
                    current_idx = img_index;
                }
                // 打包入队 (只给 AI)
                FrameData data;
                data.frame = frame_copy;
                data.index = current_idx;
                data.channel_id = i; // 【关键】贴上标签

                img_queue.enqueue(data);
                has_new_data = true; // 证明有数据读出
            }
        }
        if (!has_new_data)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // 打印日志
        if (img_index % 60 == 0)
        {
            printf("Read img_index: %d\n", img_index);
        }
    }
    finish = true;
}

// 2.处理视频
// mutex bufferMutex;// std::map<int, Mat> ProcessFrameBuffer;
// ProcessFrameBuffer 是“帧处理缓冲区”
// map 是共享资源，多线程访问必须加锁 ,所以配套bufferMutex
void Thread_ProcressVideo(SafeQueue<FrameData> &r_queue, SafeQueue<FrameData> &w_queue, bool &read_finish, bool &process_finish)
{
    // r_queue：读线程生产的帧队列（input）
    // w_queue：你准备传给写线程的处理后队列（output），但现在你还没用它
    // finished：读线程是否结束
    // A. 定义流水线深度
    // 允许同时有 16 个任务在后台跑 (建议略大于线程数 12)
    const int PIPELINE_LIMIT = 16; // PIPELINE_LIMIT：防止内存爆掉。如果不限制，读线程可能瞬间读几千帧塞进线程池，导致内存溢

    // B. 定义流水线队列，这个队列用来按顺序保存发出去的任务凭证
    std::queue<PendingTask> pipeline;
    printf("Process thread started...\n");

    while (true)
    {
        // =========================================================
        // C. 发货阶段 (Filling Pipeline)
        // 只要流水线没满，且有数据，就一直往线程池里塞
        // =========================================================

        while (pipeline.size() < PIPELINE_LIMIT)
        {
            if (r_queue.empty())
            {
                break;
            }
            FrameData frame_in;
            r_queue.dequeue(frame_in);                                                          // 取出数据，此时frame_in当作是临时存放变量 // 【关键修改】提交给线程池，拿到 Future（此时代码进入线程池内部）
            std::future<cv::Mat> fut = gthreadpool.sumbit_task(frame_in.frame, frame_in.index); // 如果pipeline.size小于16，则一直会提交任务，并且执行任务
            // 【关键修改】存入流水线使用 std::move 是因为 future 只能移动不能复制
            pipeline.push(PendingTask{frame_in.index, frame_in.channel_id, std::move(fut)}); // 把 channel_id 也存进暂存区
        }
        // =========================================================
        // D. 收货阶段 (Filling Pipeline)
        // =========================================================
        if (!pipeline.empty())
        {
            // 获取队头任务
            PendingTask &front_task = pipeline.front();
            if (front_task.fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) // 只有准备好了，才去 get()，这时候是瞬间返回的
            {
                cv::Mat res = front_task.fut.get();
                //==================================关键修改============================================
                FrameData result_back;
                result_back.frame = res;
                result_back.index = front_task.index;
                result_back.channel_id = front_task.channerl_id;

                w_queue.enqueue(result_back);
                pipeline.pop();
            }
            // 如果没准备好 (status == timeout)，代码直接往下走，
            // 也就是绕回 while(true) 的开头，重新去执行“阶段 C”进货！

            // // 获取结果，但是.get是阻塞，如果后台还没算完，这里死等
            // cv::Mat res = front_task.fut.get();
            // // 传给写队列
            // w_queue.enqueue({res, front_task.index});
            // pipeline.pop();
        }

        // =========================================================
        // E. 退出判断 (Termination Check)
        // =========================================================
        if (read_finish && r_queue.empty() && pipeline.empty())
        {
            printf("处理线程全部结束。\n");
            // 【修改点 2】设置处理结束标志，通知写线程
            process_finish = true;
            break;
        }
    }
}

// 3.创建写入视频的函数（消费者）
// void Thread_WriterVideo(cv::VideoWriter &writer, SafeQueue<FrameData> &img_q, bool &process_finish)
// {
//     // frame_tmp完整地接收从队列中取出的整个数据包（包含帧内容和索引）包含 Mat + index
//     // img_tmp用来单独存放从frame_tmp中提取出来的图像帧，以便后续处理
//     Mat img_tmp;
//     FrameData frame_tmp;
//     // start：计时起点，用于后面计算“间隔多久写一帧”。
//     auto start = std::chrono::high_resolution_clock::now();
//     while (1)
//     {

//         // =========================================================
//         // 【修改点 2】退出条件逻辑
//         // 只有当：处理线程说结束了 (process_finish) && 写队列也没货了
//         // 才是真正的结束
//         // =========================================================
//         if (process_finish && img_q.empty())
//         {
//             printf("写线程结束 (All Finished)\n");
//             break;
//         }
//         if (img_q.empty())
//         {
//             // 如果队列空了但还没 finish，就稍等一下，避免死循环空转
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             continue;
//         }

//         // ② 直接阻塞式取数据（队列空了就睡觉）
//         img_q.dequeue(frame_tmp);
//         img_tmp = frame_tmp.frame;
//         if (!img_tmp.empty())
//         {
//             // VideoWriter 的 write 本身就是阻塞的（写硬盘IO操作）
//             // 它写多快，我们就跑多快，不再人为限速
//             writer.write(img_tmp);
//         }
//         // 打印进度
//         if (frame_tmp.index > 0 && frame_tmp.index % 60 == 0)
//         {
//             printf("write index %d finished \n", frame_tmp.index);
//         }
//     }
// }

// 3.创建写入视频的函数（消费者）—— 【修改版：使用 Linux 管道】
// 注意：第一个参数改成了 FILE* pipe_fp，不再是 VideoWriter
// 3. 【最终消费者】拼图 + 推流
// 新增参数：width, height (画布的总宽高，必须和 FFmpeg 设定的 -s 参数一致)
void Thread_WriterVideo(FILE *pipe_fp, SafeQueue<FrameData> &res_queue, bool &process_finish, int width, int height)
{
    // 1. 本地缓存：存放 4 路视频的最新一帧“已推理画面”
    // 初始化为黑色背景，防止刚启动时某路没图显示花屏
    // 我们假设最多支持 4 路，每路缩放为原来的 1/2
    int sub_w = width / 2;
    int sub_h = height / 2;

    std::vector<cv::Mat> display_cache(4);
    for (int i = 0; i < 4; i++)
    {
        display_cache[i] = cv::Mat::zeros(sub_h, sub_w, CV_8UC3);
    }

    // 2. 大画布 (拼图结果)
    // 这张图的大小必须等于 main 函数里 ffmpeg 命令 -s 指定的大小
    cv::Mat mosaic_canvas(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    FrameData data_pack;
    printf("Writer Thread Started... Canvas Size: %dx%d\n", width, height);

    while (true)
    {
        // 退出条件
        if (process_finish && res_queue.empty())
        {
            printf("写线程结束 (All Finished)\n");
            break;
        }

        if (res_queue.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // 3. 取出 AI 算好的图
        res_queue.dequeue(data_pack);

        if (data_pack.frame.empty())
            continue;

        // 4. 【核心逻辑】更新本地缓存
        // 不管来的是哪一路，先把它缩放成 1/4 大小，存进对应的格子里
        cv::Mat resized_small;
        cv::resize(data_pack.frame, resized_small, cv::Size(sub_w, sub_h));

        // 安全检查：防止 ID 越界
        if (data_pack.channel_id >= 0 && data_pack.channel_id < 4)
        {
            display_cache[data_pack.channel_id] = resized_small;
        }

        // 5. 重新拼图 (每次有新图来，就重绘一遍大画布)
        // 0号放左上 (0, 0)
        display_cache[0].copyTo(mosaic_canvas(cv::Rect(0, 0, sub_w, sub_h)));
        // 1号放右上 (sub_w, 0)
        display_cache[1].copyTo(mosaic_canvas(cv::Rect(sub_w, 0, sub_w, sub_h)));
        // 2号放左下 (0, sub_h)
        display_cache[2].copyTo(mosaic_canvas(cv::Rect(0, sub_h, sub_w, sub_h)));
        // 3号放右下 (sub_w, sub_h)
        display_cache[3].copyTo(mosaic_canvas(cv::Rect(sub_w, sub_h, sub_w, sub_h)));

        // 6. 推流发送
        // 只有当管道有效时才写
        if (pipe_fp)
        {
            // 注意：这里写的是 mosaic_canvas (拼好的大图)，而不是 data_pack.frame
            fwrite(mosaic_canvas.data, 1, mosaic_canvas.total() * mosaic_canvas.elemSize(), pipe_fp);
        }

        // 打印进度 (每30帧打印一次，避免刷屏)
        if (data_pack.index > 0 && data_pack.index % 30 == 0)
        {
            printf("Pushed Mosaic Frame %d (from Cam %d)\n", data_pack.index, data_pack.channel_id);
        }
    }
}

//(老师原版)定义一个写入视频帧的线程函数
// void Thread_WriterVideo(VideoWriter &writer, SafeQueue<FrameData> &img_queue, bool &finished)
// {
//     Mat img_temp;
//     FrameData frame_temp;
//     auto start = std::chrono::high_resolution_clock::now();
//     // 无限循环，直到写入结束
//     while (true)
//     {
//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<chrono::milliseconds>(end - start);
//         // 如果写入队列不为空且时间间隔超过30毫秒，则取出一帧进行写入
//         if (!img_queue.empty())
//         {
//             if (duration.count() > 30)
//             {
//                 img_queue.dequeue(frame_temp);
//                 img_temp = frame_temp.frame;
//                 if (!img_temp.empty())
//                 {
//                     start = std::chrono::high_resolution_clock::now();
//                     end = std::chrono::high_resolution_clock::now();
//                     writer.write(img_temp);
//                 }
//                 // 每写入100帧打印一次信息
//                 if (frame_temp.index % 10 == 0)
//                 {
//                     printf("write index %d finished!\r\n", frame_temp.index);
//                 }
//             }
//         }
//         else if (finished)
//         {
//             // 如果写入结束标志被设置，则打印写入结束信息并退出循环
//             printf("write end\r\n");
//             break;
//         }
//     }
// }

//=============简单的画框函数===========================
void draw_detections(
    cv::Mat &img,
    const std::vector<Detection> &dets,
    const std::vector<std::string> &class_names) // 类别名字，可选
{
    for (const auto &det : dets)
    {
        // 1. 构造矩形框（左上角 + 宽高）
        cv::Rect rect;
        rect.x = static_cast<int>(det.x1);
        rect.y = static_cast<int>(det.y1);
        rect.width = static_cast<int>(det.x2 - det.x1);
        rect.height = static_cast<int>(det.y2 - det.y1);

        // 2. 画框
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2); // 绿色框，线宽 2

        // 3. 准备一行文字：类别名 + 分数
        std::string label;
        if (!class_names.empty() && det.class_id >= 0 && det.class_id < (int)class_names.size())
        {
            label = class_names[det.class_id];
        }
        else
        {
            label = std::to_string(det.class_id);
        }
        label += cv::format(" %.2f", det.score);

        // 4. 文字位置（框的左上角上面一点）
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = rect.x;
        int y = std::max(rect.y - 5, label_size.height);

        // 画一个实心矩形作为文字背景
        cv::rectangle(
            img,
            cv::Point(x, y - label_size.height),
            cv::Point(x + label_size.width, y + baseLine),
            cv::Scalar(0, 255, 0),
            cv::FILLED);

        // 再把文字画上去
        cv::putText(
            img, label,
            cv::Point(x, y),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 0, 0), 1);
    }
}

GB28181Agent g_sip_agent; // 全局国标代理实例

// 主函数
int main()
{
    // 初步调试GB28181 LiveGBS地址：http://172.18.194.174:8080/#/devices/1?online=true
    const char *server_ip = "192.168.137.1";        // 你的国标平台 IP (需修改)
    const char *device_id = "34020000001320000001"; // 设备 20 位 ID（可以自己选择合适的）
    const char *password = "gbs12345";              // 注册密码
    cout << ">>> 正在启动国标 GB28181 信令模块..." << endl;
    // 平台 SIP 端口：15060
    // 本机监听端口：5061
    // 本机监听IP：0.0.0.0 可以（监听全部网卡）
    if (g_sip_agent.start(server_ip, 15060, "0.0.0.0", 5060, device_id, password) != 0)
    {
        cerr << "!!! 国标代理启动失败，请检查网络或端口占用" << endl;
        // 即使失败，通常也建议继续运行本地 AI 逻辑
    }

    // while (1); // 【已修复】移除了这里的死循环，让代码能继续向下执行

    // ================= 修改开始 =================
    // 1. 设置 多路RTSP 拉流地址
    // 这里必须填你【虚拟机】的 IP (192.168.137.181) 和端口 (8554)
    // 路径必须和你 FFmpeg 推流命令里的 /live/test 一致
    std::vector<string> rtsp_url = {
        "rtsp://192.168.137.181:8554/live/test",
        "rtsp://192.168.137.181:8554/live/test",
        "rtsp://192.168.137.181:8554/live/test",
        "rtsp://192.168.137.181:8554/live/test"};
    // 批量创建StreamLoader对象
    // 注意：StreamLoader 构造函数里还没开始连网
    std::vector<std::shared_ptr<StreamLoader>> cameras;
    for (int i = 0; i < rtsp_url.size(); i++)
    {
        // A创建对象
        auto loader = std::make_shared<StreamLoader>(rtsp_url[i], i); // 后面是传递给构造函数的参数
        cameras.push_back(loader);                                    // 放入vector中
        // B启动线程
        std::thread t([loader]()
                      { (*loader)(); });
        // 线程分离
        t.detach();
        printf("Camera [%d] started...\n", i);
    }

    // 3. 【更稳健的等待】等待所有摄像头都准备好
    // 我们必须确保每一路都有图了，才能进入后面的主循环，否则拼图时 resize 空图会崩
    printf("Waiting for all streams to initialize...\n");

    for (int i = 0; i < cameras.size(); i++)
    {
        while (true)
        {
            // 加锁检查（因为后台线程正在写这个 mat）
            // 虽然 empty() 检查通常很快，但为了严谨最好加锁，或者你的类里有个 is_ready 原子变量更好
            bool is_empty = false;
            is_empty = cameras[i]->bgrMat.empty();

            if (!is_empty)
            {
                printf("Camera [%d] is ready!\n", i);
                break; // 这一路好了，跳出 while，去检查下一路（i++）
            }

            // 还没好，睡一会再看
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            // printf("Waiting for Camera [%d]...\n", i); // 可以在这里打印等待日志
        }
    }
    printf("All cameras are online! Starting main loop...\n");

    // 获取分辨率 (直接从 bgrMat 获取，或者你在 StreamLoader 里加个 getWidth 接口)
    // 如果 bgrMat 还是空的，说明还没连上，先给个默认值防止报错
    int width = cameras[0]->bgrMat.cols;
    int height = cameras[0]->bgrMat.rows;

    printf("Camera init success (Wait for first frame)... size: %dx%d\n", width, height);

    // ========= 初始化模型 (保持不变) ==================
    post_process(); // 会把 labels_vector 填好
    // // 测试图像

    mutex cap_m;

    // 利用容器创建多线程，创建读视频的线程
    int img_index = -1;
    int num_thread = 1;

    // 标志1：读完了吗？ (给 Reader 改，Process 看)
    bool is_read_done = false;
    // 标志2：处理完了吗？ (给 Process 改，Writer 看)
    bool is_process_done = false;

    std::vector<thread> video_readers;
    for (int i = 0; i < num_thread; i++)
    {
        video_readers.emplace_back(Thread_ReadVideo,
                                   ref(cameras), // 传 StreamLoader 引用
                                   ref(SafeQueue_Read),
                                   ref(img_index),
                                   ref(cap_m),
                                   ref(is_read_done));
    }

    // 写入视频
    // cv::Size frame_size(width, height);
    // // 注意：USB摄像头无法像文件那样直接获取准确 FPS，通常设为 30 或 25
    // double fps = 30.0;
    // cv::VideoWriter writer("/home/orangepi/opencv_test/output_v4l2.avi", cv::VideoWriter::fourcc('I', '4', '2', '0'), fps, frame_size);
    // 写入视频

    // ================= 核心修改开始 =================
    // 写入视频配置
    cv::Size frame_size(width, height);
    double fps = 30.0;

    // ================= 核心修改开始 =================
    // 1. 设置推流地址 (确认是你刚才验证成功的那个 IP)
    std::string rtmp_url = "rtmp://192.168.137.181/live/result";

    // 2. 组装 FFmpeg 命令管道 (和之前一样)
    // 注意： -s 1280x720 必须和你前面定义的 width/height 完全一致
    std::string command = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height) + " -r " + std::to_string(fps) + " -i - -c:v libx264 -pix_fmt yuv420p -preset ultrafast -f flv " + rtmp_url;

    // 3. 【重点】使用 Linux 系统调用 popen 打开一个进程
    // "w" 表示我们要向这个进程“写”数据
    // 这相当于在代码里帮你运行了刚才那条 ffmpeg 命令
    FILE *ffmpeg_pipe = popen(command.c_str(), "w");

    if (ffmpeg_pipe == nullptr)
    {
        std::cerr << "无法打开 FFmpeg 管道，请检查命令！" << std::endl;
        return -1;
    }
    // ================= 核心修改结束 =================

    // 创建一个处理的线程
    std::thread video_p(Thread_ProcressVideo, ref(SafeQueue_Read), ref(SafeQueue_Write), ref(is_read_done), ref(is_process_done));

    // 创建一个写入视频的线程
    // 创建一个写入视频的线程 —— 【注意：这里传参变了】
    // 传入的是 ffmpeg_pipe 指针，而不是 writer
    std::thread video_w(Thread_WriterVideo, ffmpeg_pipe, ref(SafeQueue_Write), ref(is_process_done), width, height);

    // 回收线程资源
    for (thread &t : video_readers)
    {
        t.join();
    }

    video_p.join(); // 等待处理视频线程完成
    video_w.join(); // 等待写入视频线程完成

    // 【新增】程序结束前，关闭管道
    pclose(ffmpeg_pipe);
    return 0;
}
