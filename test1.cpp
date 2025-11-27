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
std::map<int, Mat> ProcessFrameBuffer;
// ProcessFrameBuffer 是“帧处理缓冲区”
mutex bufferMutex;
// map 是共享资源，多线程访问必须加锁,所以配套bufferMutex
void Thread_ProcressVideo(SafeQueue<FrameData> &r_queue, SafeQueue<FrameData> &w_queue, bool &finish)
{
    // r_queue：读线程生产的帧队列（input）
    // w_queue：你准备传给写线程的处理后队列（output），但现在你还没用它
    // finished：读线程是否结束
    FrameData frame_temp;
    int next_index = 0;
    while (true)
    {
        if (finish && r_queue.empty())
        {
            printf("process end, total processed = %d\n", next_index);
            printf("process end\n");
            break;
        }
        // （A）第一部分：从 r_queue 取出帧 → 临时放入 map
        // dequeue() 会安全地把一帧拿出来放入 frame_temp
        r_queue.dequeue(frame_temp); // 从刚刚读取视频的队列中取出一帧放入frame_temp里面，dequeue是安全的（不需要再去判断队列是否
        {
            // 将帧写入处理缓冲区
            lock_guard<mutex> lock(bufferMutex); // 一定要枷锁，map<int, Mat> 是共享容器，多线程同时写 map 会崩溃
            ProcessFrameBuffer[frame_temp.index] = frame_temp.frame.clone();
        }
        // （B）第二部分：从 map 里按顺序处理帧
        // 先判断要处理的下一帧（按顺序）图像是否放在缓冲里

        while (!ProcessFrameBuffer.empty() && ProcessFrameBuffer.count(next_index))
        {
            Mat img;
            auto it = ProcessFrameBuffer.find(next_index);
            if (it != ProcessFrameBuffer.end())
            {
                img = it->second;

                // -----处理图像待完成
                gthreadpool.sumbit_task(img.clone(), g_frame_start_id++); // 在这里next_index不是0？第一次循环确实是0，但是后面又++

                gthreadpool.get_result(img, next_index);
                //------------

                // 入队
                w_queue.enqueue({img, next_index});
                ProcessFrameBuffer.erase(it);
                if (next_index > 0 && next_index % 10 == 0)
                {
                    printf("process index %d finished \n", next_index);
                }
                next_index++;
            }
        }
    }
}

// 3.创建写入视频的函数（消费者）
void Thread_WriterVideo(cv::VideoWriter &writer, SafeQueue<FrameData> &img_q, bool &finish)
{
    // frame_tmp完整地接收从队列中取出的整个数据包（包含帧内容和索引）包含 Mat + index
    // img_tmp用来单独存放从frame_tmp中提取出来的图像帧，以便后续处理
    Mat img_tmp;
    FrameData frame_tmp;
    // start：计时起点，用于后面计算“间隔多久写一帧”。
    auto start = std::chrono::high_resolution_clock::now();
    while (1)
    {

        // 先判断是否结束（队列为空，且标志位为True）
        if (finish && img_q.empty())
        {
            printf("all finished\n");
            break;
        }
        // ② 直接阻塞式取数据（队列空了就睡觉）
        img_q.dequeue(frame_tmp);
        /*每次循环都拿当前时间 end计算从 start 到现在 end 经过了多少毫秒 duration,
       用来决定是不是“到了该写下一帧的时候”*/
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (duration.count() < 30)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(30 - duration.count()));
        }

        start = std::chrono::high_resolution_clock::now(); // 重新记录写入时间起点，为下一次计时做准备。
        end = std::chrono::high_resolution_clock::now();

        // 写入帧
        img_tmp = frame_tmp.frame;
        if (!img_tmp.empty()) // 防御性检查确保这一帧不是空图像
        {
            // 如果队列第一个图片不是空，就写入到VideoWriter这个类里面（对象是writer）
            writer.write(img_tmp);
        }
        // 每写10次循环
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

    Yolov5s yolov5s("/home/orangepi/opencv_test/model/yolov5s.rknn", 0);
    while (1);
    // 定义锁(全局)，专门用于在读取原视频的时候，锁住，防止多个线程读取原视频
    mutex cap_m;

    // 利用容器创建多线程，创建读视频的线程
    int img_index = -1;
    int num_thread = 1;
    bool finished = false; // 判断读取是否结束
    std::vector<thread> video_readers;
    for (int i = 0; i < num_thread; i++)
    {
        video_readers.emplace_back(Thread_ReadVideo, ref(cap), ref(SafeQueue_Read), ref(img_index), ref(cap_m), ref(finished));
    }

    // 写入视频
    cv::Size frame_size(width, height);
    cv::VideoWriter writer("/home/orangepi/opencv_test/output.avi", cv::VideoWriter::fourcc('I', '4', '2', '0'), fps, frame_size);

    // 创建一个处理的线程
    std::thread video_p(Thread_ProcressVideo, ref(SafeQueue_Read), ref(SafeQueue_Write), ref(finished));

    // 创建一个写入视频的线程
    std::thread video_w(Thread_WriterVideo, ref(writer), ref(SafeQueue_Write), ref(finished));

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
