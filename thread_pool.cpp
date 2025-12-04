#include "thread_pool.h"

ThreadPool::ThreadPool(int num_thread) : run(true)
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
//----------------插入任务函数
std::future<cv::Mat> ThreadPool::sumbit_task(const cv::Mat &img, int index)
{
    // 1. 定义具体的任务逻辑 (Lambda)
    // 这里的代码将来会在 Worker 线程里执行
    // [img, index]：把图片和序号捕获（复制）进这个任务包里
    // “定义一个名字叫 job_func 的匿名任务。这个任务随身携带了外面的 img 和 index 的副本（背包）。启动这个任务不需要传参（括号是空的）。任务做完后，承诺会吐出一张 cv::Mat 图片（箭头指向返回值）。
    auto job_func = [img, index]() -> cv::Mat
    {
        // --- 模拟耗时操作 (YOLO推理) ---
        printf("  [Worker] 开始处理第 %d 帧...\n", index);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        cv::Mat res = img.clone(); // 这里的 clone 很重要，保证结果是独立的?????？
        return res;
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
        tasks.push(std::move(task)); 

    }
    // 5. 通知
    task_cond.notify_one();
    // 6. 返回票
    return res_future;
}



//----------------工作者函数------------------
void ThreadPool::worker(int id)
{
    while (run)
    {
        // 定义一个空的任务包用来接货
        std::packaged_task<cv::Mat()> current_task;

        {
            //任务时先上锁
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
        // 接下来是对取出来的任务进行处理
        //===================
        printf("worker 正在工作ing\n");
        std::this_thread::sleep_for(std::chrono::seconds(2));
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
