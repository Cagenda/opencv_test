#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <condition_variable>
#include<atomic>
#include<thread>

// 1. 包含OpenCV核心功能头文件，定义了  cv::Mat
#include <opencv2/highgui.hpp>
#include <future>      // 【新增】核心库：用于支持 future 和 promise
#include <functional>  // 【新增】核心库：用于支持 std::function
class ThreadPool
{
public:
    ThreadPool(int num_thread);//num_threads是线程池数量
    ~ThreadPool();

    // 【修改】返回值变成了 std::future<cv::Mat> 这意味着：提交任务后，你会立刻得到一个“未来的结果凭证”
    std::future<cv::Mat> sumbit_task(const cv::Mat &img, int index);
    
    // 【删除】删除了 get_result 函数，因为现在凭证在调用者手里，不需要去池子里找
    // int get_result(cv::Mat &img, int index);//这个函数是用来干什么？


private:
    std::atomic<bool> run;//用来判断线程池是否关闭
    
    //任务队列里直接存 packaged_task
    std::queue<std::packaged_task<cv::Mat()>>tasks;
   
    std::mutex task_mtx;
    std::condition_variable task_cond;//在任务被放入任务队列后，用来通知worker来取任务了

    // 【删除】删除了 img_result, res_mtx, res_cond
    // 因为结果不再保存在线程池内部
    // std::map<int, cv::Mat> img_result;//用来存放处理之后的结果
    // std::mutex res_mtx;
    // std::condition_variable res_cond; 


    //线程池
    std::vector<std::thread> threads;
    //工作者函数
    void worker(int id);
    
    
};// 用来通知“有结果了”,可以从结果的map中取出结果

#endif