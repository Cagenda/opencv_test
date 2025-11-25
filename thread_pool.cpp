#include"thread_pool.h"

ThreadPool::ThreadPool(int num_thread):run(true)
{

}
ThreadPool::~ThreadPool()
{
    std::cout << "析构线程池" <<std::endl;
    this->run = false;
}

int ThreadPool::init(int num_thread)
{
    for (size_t i = 0; i < num_thread; i++)
    {
        //创建了工作者线程
        std::thread t(&ThreadPool::worker, this, i);
    }
    std::cout << "ThreadPool Init" << std::endl;
    return 0;
}
//----------------插入任务函数
int ThreadPool::sumbit_task(const cv::Mat &img, int index)
{
    {
        std::lock_guard<std::mutex> lock(task_mtx);
        tasks.push({index, img});
    }
    task_cond.notify_one();
    return 0;
}

//----------------工作者函数
void ThreadPool::worker(int id)
{
    while (run)
    {
        std::unique_lock<std::mutex> lock(task_mtx);
        task_cond.wait(lock, [this]{ return (!tasks.empty() || !run); });
        if(!run)
        {
           std::cout << "worker %d 下班" << id << std::endl;
           return;//break也行吗？
        }

        //如果任务队列不为空
        std::pair<int, cv::Mat> task;//临时变量接收
        task = tasks.front();
        tasks.pop();

        //接下来是对取出来的任务进行处理
        //===================


        //===================
        //任务完成之后插入结果
        {
            std::lock_guard<std::mutex> lock(res_mtx);
            img_result.insert(std::make_pair(task.first,task.second));
        }
        //在插入结果后，通知（获取结果函数）
        res_cond.notify_one();
    }
}

//----------------获取结果函数
int ThreadPool::get_result(cv::Mat &img, int index)
{
using namespace std::chrono;
//=============改进版本===========================
 std::unique_lock<std::mutex> lock(res_mtx);
 int loop = 0;
const int max_loop = 1000;                  // 和原来逻辑对应：最多等待 1000 次
const auto duration = milliseconds(5);          // 每次最多等 5ms

while (img_result.find(index) == img_result.end())
{
    // 如果线程池已经停了，而且没有这个结果，就不用再等了
    if (!run)
    {
            std::cout << "ThreadPool 已停止，index " << index
                      << " 没有结果" << std::endl;
            return -1;
    }

// 等待最多 5ms，有新结果插入时 res_cond.notify_one() 会唤醒/*等结果的线程醒来后执行的判断：如果我要找的 index 结果已经出现在 img_result 里，或者线程池要退出了那么停止等待，结束 wait_for。*/
    res_cond.wait_for(
            lock,
            duration,
            [this, index]
            {

                return (img_result.find(index) != img_result.end()) || !run;
            });
  // 2️⃣ 5ms 结束 / 被 notify / run=false / 条件变真 之后，程序就是从这里继续往下执行

        ++loop;
        if (loop > max_loop )
        {
            std::cout << "Get results Timeout for index " << index << std::endl;
            return -1; // 超时：返回错误码
        }
    }

 // 能走到这里说明 img_result[index] 一定存在，并且还在持有 res_mtx
    auto it = img_result.find(index);
    img = it->second;           // 拷贝/浅拷贝图像
    img_result.erase(it);       // 删除这条记录，防止内存堆积
    return 0;


}

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
