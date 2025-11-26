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

class ThreadPool
{
public:
    ThreadPool(int num_thread);//num_threads是线程池数量
    ~ThreadPool();
    int sumbit_task(const cv::Mat &img, int index);
    int get_result(cv::Mat &img, int index);//这个函数是用来干什么？


private:
    std::atomic<bool> run;//用来判断线程池是否关闭
    std::queue<std::pair<int, cv::Mat>> tasks;//定义任务队列

    std::mutex task_mtx;
    std::condition_variable task_cond;//在任务被放入任务队列后，用来通知worker来取任务了

    std::map<int, cv::Mat> img_result;//用来存放处理之后的结果
    std::mutex res_mtx;

    //线程池
    std::vector<std::thread> threads;
    //工作者函数
    void worker(int id);
    
    std::condition_variable res_cond; 
};// 用来通知“有结果了”,可以从结果的map中取出结果

#endif