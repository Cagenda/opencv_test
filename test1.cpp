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
#include "thread_pool.h"
// 2. 包含图像I/O和GUI函数头文件， 定义了 imread, imshow, waitKey#include <iostream>
using namespace std;
using namespace cv;

// 定义一个线程池
ThreadPool gthreadpool(12);
static int g_frame_start_id = 0; // 用于帧的起始id

// 定义每一帧（也就是每一张的图片）的信息
struct FrameData
{
    cv::Mat frame; // 创建一个Mat类型的图片（也就是一帧）
    int index;     // 帧索引
};
// 定义“流水线任务”结构体：我们需要一个结构体来暂存“发出去的订单”。
struct PendingTask
{
    int index;                // 帧序号
    std::future<cv::Mat> fut; // 取餐票 (注意这里是 future<Mat>)
};

// 创建读取视频队列
SafeQueue<FrameData> SafeQueue_Read;

// 创建写入视频的队列
SafeQueue<FrameData> SafeQueue_Write;

// 1. 【生产者】线程函数：只能有一个线程执行它（有mutex）
// 它的职责是顺序读取视频，并安全地将帧放入队列
void Thread_ReadVideo(VideoCapture &video, SafeQueue<FrameData> &img_queue, int &img_index, mutex &cap_mutex, bool &finish)
{
    while (1)
    {
        FrameData frame_tmp;
        {
            std::lock_guard<mutex> cap_lock(cap_mutex); // 加锁访问读取数据
            // 尝试从共享的video中读取下一帧，并将图像数据直接存入托盘的.frame部分。
            if (!video.read(frame_tmp.frame)) // 这里的video.read()已经读取了。读取内容放到了frame_tmp.frame
            {
                break;
            }
        }
        // 读取到图片
        img_index++;
        frame_tmp.index = img_index;
        img_queue.enqueue(frame_tmp); // 安全入队
        if (img_index > 0 && img_index % 10 == 0)
        {
            printf("read img_index:%d:\n", img_index);
        }
    }
    finish = true;
    printf("read end:\n");
}

// 2.处理视频
// mutex bufferMutex;// std::map<int, Mat> ProcessFrameBuffer;
// ProcessFrameBuffer 是“帧处理缓冲区”
// map 是共享资源，多线程访问必须加锁,所以配套bufferMutex
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
            r_queue.dequeue(frame_in); // 取出数据
                                       // 【关键修改】提交给线程池，拿到 Future原来是 get_result 死等，现在是 submit_task 立刻拿票走人
            std::future<cv::Mat> fut = gthreadpool.sumbit_task(frame_in.frame, frame_in.index);
            // 【关键修改】存入流水线使用 std::move 是因为 future 只能移动不能复制
            pipeline.push({frame_in.index, std::move(fut)});
            printf("已提交帧: %d\n", frame_in.index);
        }
        // =========================================================
        // D. 收货阶段 (Filling Pipeline)
        // =========================================================
        if (!pipeline.empty())
        {
            // 获取队头任务
            PendingTask &front_task = pipeline.front();

            // 获取结果，但是.get是阻塞，如果后台还没算完，这里死等
            cv::Mat res = front_task.fut.get();
            // 传给写队列
            w_queue.enqueue({res, front_task.index});
            pipeline.pop();
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
void Thread_WriterVideo(cv::VideoWriter &writer, SafeQueue<FrameData> &img_q, bool &process_finish)
{
    // frame_tmp完整地接收从队列中取出的整个数据包（包含帧内容和索引）包含 Mat + index
    // img_tmp用来单独存放从frame_tmp中提取出来的图像帧，以便后续处理
    Mat img_tmp;
    FrameData frame_tmp;
    // start：计时起点，用于后面计算“间隔多久写一帧”。
    auto start = std::chrono::high_resolution_clock::now();
    while (1)
    {

        // =========================================================
        // 【修改点 2】退出条件逻辑
        // 只有当：处理线程说结束了 (process_finish) && 写队列也没货了
        // 才是真正的结束
        // =========================================================
        if (process_finish && img_q.empty())
        {
            printf("写线程结束 (All Finished)\n");
            break;
        }
        if (img_q.empty())
        {
            // 如果队列空了但还没 finish，就稍等一下，避免死循环空转
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ② 直接阻塞式取数据（队列空了就睡觉）
        img_q.dequeue(frame_tmp);
        img_tmp = frame_tmp.frame;
        if (!img_tmp.empty())
        {
            // VideoWriter 的 write 本身就是阻塞的（写硬盘IO操作）
            // 它写多快，我们就跑多快，不再人为限速
            writer.write(img_tmp);
        }
        // 打印进度
        if (frame_tmp.index > 0 && frame_tmp.index % 10 == 0)
        {
            printf("write index %d finished \n", frame_tmp.index);
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

// 主函数
int main()
{
    // 记录时间
    auto start = std::chrono::steady_clock::now();

    // 测试视频
    char video_path[] = "1.mp4";
    cv::VideoCapture cap(video_path);
    // 打开错误判断
    if (!cap.isOpened())
    {
        perror("Video_unopened");
        return -1;
    }
    // 获取视频的长宽，以及帧数
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    int frame_num = cap.get(CAP_PROP_FRAME_COUNT);

    printf("size:%d height:%d fps:%f total:%d\n", width, height, fps, frame_num);
    cv::Mat img_tmp;
    cap.read(img_tmp);
    if (img_tmp.empty())
    {
        perror("img_tmp failed");
    }
    Yolov5s yolov5s("/home/orangepi/opencv_test/model/yolov5s.rknn", 0);
    yolov5s.inference_image(img_tmp);

    // 定义锁(全局)，专门用于在读取原视频的时候，锁住，防止多个线程读取原视频
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
        video_readers.emplace_back(Thread_ReadVideo, ref(cap), ref(SafeQueue_Read), ref(img_index), ref(cap_m), ref(is_read_done));
    }

    // 写入视频
    cv::Size frame_size(width, height);
    cv::VideoWriter writer("/home/orangepi/opencv_test/output.avi", cv::VideoWriter::fourcc('I', '4', '2', '0'), fps, frame_size);

    // 创建一个处理的线程
    std::thread video_p(Thread_ProcressVideo, ref(SafeQueue_Read), ref(SafeQueue_Write), ref(is_read_done), ref(is_process_done));

    // 创建一个写入视频的线程
    std::thread video_w(Thread_WriterVideo, ref(writer), ref(SafeQueue_Write), ref(is_process_done));

    // 回收线程资源
    for (thread &t : video_readers)
    {
        t.join();
    }

    video_p.join(); // 等待处理视频线程完成
    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "程序总运行时间：" << dur.count() << " ms\n";
    video_w.join(); // 等待写入视频线程完成
    return 0;
}
