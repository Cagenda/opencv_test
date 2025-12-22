#include "thread_pool.h"
static thread_local std::unique_ptr<Yolov5s> tls_yolo; // tls_yolo 是一个智能指针，指向一个 Yolov5s 对象。
// thread_local 表示：这个变量对“每个线程”来说都是独立的副本。
static thread_local int tls_worker_id = -1; // 每一个worker线程，都有自己独立的tls_worker_id

// 谁第一次用到就初始化（发生在任务执行线程里）让 worker() 和 submit_task() 里的 lambda 都能访问到同一份 TLS 变量（同一线程内同一份）
static Yolov5s &get_tls_yolo(const std::string &model_path, int npu_core_num)
{
    if (!tls_yolo)
    {
        int core = (npu_core_num > 0 && tls_worker_id >= 0) ? (tls_worker_id % npu_core_num) : 0;
        printf("[init] worker=%d core=%d\n", tls_worker_id, core);
        tls_yolo = std::unique_ptr<Yolov5s>(new Yolov5s(model_path.c_str(), core)); // 这一步才会真正调用 Yolov5s 构造函数。在yolo构造函数中，为这个线程设置NPU的核
    }
    return *tls_yolo;
}

//=================================画框函数==============================================
static void draw_detections(cv::Mat &img,
                            const std::vector<Detection> &dets,
                            const std::vector<std::string> &labels)
{
    for (const auto &det : dets)
    {
        cv::rectangle(img,
                      cv::Point((int)det.x1, (int)det.y1),
                      cv::Point((int)det.x2, (int)det.y2),
                      cv::Scalar(0, 255, 0), 2);

        std::string name = (det.class_id >= 0 && det.class_id < (int)labels.size())
                               ? labels[det.class_id]
                               : std::to_string(det.class_id);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), det.score);

        int baseLine = 0;
        cv::Size tsize = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        int x = std::max(0, (int)det.x1);
        int y = std::max(0, (int)det.y1 - tsize.height - 5);

        cv::rectangle(img, cv::Rect(x, y, tsize.width + 6, tsize.height + baseLine + 6),
                      cv::Scalar(0, 255, 0), -1);
        cv::putText(img, buf, cv::Point(x + 3, y + tsize.height + 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

// ===============================初始化线程池==================================
ThreadPool::ThreadPool(int num_thread, const std::string &model_path, int npu_core_num) : run(true), model_path_(model_path), npu_core_num_(npu_core_num)
{
    printf("初始化线程池\n");
    for (size_t i = 0; i < num_thread; i++)
    {
        // 创建了工作者线程

        threads.emplace_back(&ThreadPool::worker, this, i);
    }
    std::cout << "ThreadPool Init" << std::endl;
}

ThreadPool::~ThreadPool()
{
    std::cout << "析构线程池" << std::endl;
    this->run = false;
    task_cond.notify_all();
    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
}
//==================================插入任务函数========================
std::future<cv::Mat> ThreadPool::sumbit_task(const cv::Mat &img, int index)
{
    // 1. 定义具体的任务逻辑 (Lambda)
    // 这里的代码将来会在 Worker 线程里执行
    // [img, index]：把图片和序号捕获（复制）进这个任务包里
    // “定义一个名字叫 job_func 的匿名任务。这个任务随身携带了外面的 img 和 index 的副本（背包）。启动这个任务不需要传参（括号是空的）。任务做完后，承诺会吐出一张 cv::Mat 图片（箭头指向返回值）。
    auto job_func = [img, index, this]() -> cv::Mat
    {
        // ✅ 这句会在“执行该任务的 worker 线程”里运行
        // 第一次调用会初始化本线程的 Yolov5s；后续复用
        Yolov5s &yolo = get_tls_yolo(this->model_path_, this->npu_core_num_); // 如果本线程是第一层被调用，则会初始化yolo的构造函数，在yolo构造函数中，为这个线程设置NPU的核

        cv::Mat out = img.clone();
        std::vector<Detection> dets;
        yolo.inference_image(out, dets);           // ✅ 真正跑一帧推理
        draw_detections(out, dets, labels_vector); // 画框
        return out;
    };
    // 用你提供的参数构造一个 T 对象返回一个 shared_ptr<T> 指向这个对象用 job_func 去构造一个 packaged_taskcv::Mat()，然后把这个任务对象用 shared_ptr 管起来

    // std::make_shared<T>(参数...)
    // std::packaged_task<返回值类型(参数列表)>

    // auto task_ptr = std::make_shared<std::packaged_task<cv::Mat()>>(job_func);
    // 2. 直接在栈上创建任务包 (不需要 make_shared)
    std::packaged_task<cv::Mat()> task(job_func);
    //-----获取Future，这一步仅仅只是链接future和promise-----------
    // 3. 拿到取餐票
    std::future<cv::Mat> res_future = task.get_future();

    // -----------真正的执行任务应该在worker线程---------------------
    // -----------把task放入队列--------------------------
    {
        // 【核心难点】必须使用 std::move() ！！！
        // 因为 task 是独占的，你必须把它“移”进队列，原来的 task 变量就空了
        std::lock_guard<std::mutex> lock(task_mtx);
        tasks.push(std::move(task)); // 将task(job_func)放进了tasks的队列中，，等待线程认领
        if (index % 30 == 0)
        {
            printf("已经提交%d个任务给线程池中的任务队列\n", index);
        }
    }
    // 5. 通知
    task_cond.notify_one();
    // 6. 返回票
    return res_future;
}

//============================工作者函数worker()===================================
void ThreadPool::worker(int id)
{
    tls_worker_id = id; // ✅ 让本线程知道自己是哪个 worker（用于绑定 core),我一共创建了6个工作者线程
    // worker(id) 启动时：tls_worker_id = id;get_tls_yolo() 用 tls_worker_id % npu_core_num 选 core

    while (run)
    {
        // 定义一个空的任务包用来接货
        std::packaged_task<cv::Mat()> current_task;
        {
            // 任务时先上锁
            std::unique_lock<std::mutex> lock(task_mtx);
            task_cond.wait(lock, [this]
                           { return (!tasks.empty() || !run); });
            if (!run)
            {
                std::cout << "worker %d 下班" << id << std::endl;
                return; // break也行吗？
            }
            // 如果任务队列不为空【核心难点】必须使用 std::move() ！！！把队列头的任务“移”到 current_task 变量里
            current_task = std::move(tasks.front());
            tasks.pop();
        }

        //===================
        // 执行任务
        // 运行完后，结果会自动蹦到主线程的 future 里
        current_task();
    }
}

//----------------获取结果函数
// int ThreadPool::get_result(cv::Mat &img, int index)
// {
//     using namespace std::chrono;
//     //=============改进版本===========================
//     std::unique_lock<std::mutex> lock(res_mtx);
//     int loop = 0;
//     const int max_loop = 1000;             // 和原来逻辑对应：最多等待 1000 次
//     const auto duration = milliseconds(5); // 每次最多等 5ms

//     while (img_result.find(index) == img_result.end())
//     {
//         // 如果线程池已经停了，而且没有这个结果，就不用再等了
//         if (!run)
//         {
//             std::cout << "ThreadPool 已停止，index " << index
//                       << " 没有结果" << std::endl;
//             return -1;
//         }

//         // 等待最多 5ms，有新结果插入时 res_cond.notify_one() 会唤醒/*等结果的线程醒来后执行的判断：如果我要找的 index 结果已经出现在 img_result 里，或者线程池要退出了那么停止等待，结束 wait_for。*/
//         res_cond.wait_for(
//             lock,
//             duration,
//             [this, index]
//             {
//                 return (img_result.find(index) != img_result.end()) || !run;
//             });
//         // 2️⃣ 5ms 结束 / 被 notify / run=false / 条件变真 之后，程序就是从这里继续往下执行

//         ++loop;
//         if (loop > max_loop)
//         {
//             std::cout << "Get results Timeout for index " << index << std::endl;
//             return -1; // 超时：返回错误码
//         }
//     }

//     // 能走到这里说明 img_result[index] 一定存在，并且还在持有 res_mtx
//     auto it = img_result.find(index);
//     img = it->second;     // 拷贝/浅拷贝图像
//     img_result.erase(it); // 删除这条记录，防止内存堆积
//     return 0;
// }

// int loop = 0;
// while(img_result.find(index)==img_result.end())
// {
// std::this_thread::sleep_for(std::chrono::milliseconds(5));
// loop++;
// if(loop>1000)
// {
//     std::cout << "Get results Timeout" <<std:: endl;
// }
// }
// {
// std::lock_guard<std::mutex> lock(res_mtx);
// img = img_result[index];
// img_result.erase(index);
// }
// return 0;
