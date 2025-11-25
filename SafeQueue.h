#ifndef _SAFEQUEUE_H_
#define _SAFEQUEUE_H_

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace std;
// 此处的T相当于一个类型
template <typename T>
class SafeQueue
{
public:
    SafeQueue() {};
    // 析构函数
    ~SafeQueue() {};
    // 插入队列
    void enqueue(const T &t)
    {
        std::lock_guard<mutex> lock(m);
        q.push(t);
        c.notify_one(); // 提醒消费者
    }
    // 出队
    void dequeue(T &t)
    {
        std::unique_lock<mutex> lock(m);
        // unique_lock 是 wait 必须要用的锁类型（可解锁/上锁）
        c.wait(lock, [this]
               { return !q.empty(); });
        /*检查队列是否为空
        如果为空：

        unlock m

        当前线程 sleep

        等待 enqueue 调用 c.notify_one()

        被唤醒后自动 lock m

        再次检查队列 non-empty 的条件

        条件成立后跳出 wait*/
        t = q.front();
        q.pop();
    }
    bool empty()
    {
        std::lock_guard<mutex> lock(m);
        return q.empty();
    }

private:
    queue<T> q;
    // mutex 是锁，保护数据安全
    mutable mutex m;
    // condition_variable  是信号，协调线程执行顺序
    condition_variable c;
};
#endif