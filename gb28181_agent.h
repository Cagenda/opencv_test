#ifndef GB28181_AGENT_H
#define GB28181_AGENT_H

#include <eXosip2/eXosip.h>
#include <osip2/osip_mt.h>
#include <string>
#include <thread>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

class GB28181Agent
{
public:
    GB28181Agent();
    ~GB28181Agent();

    // 初始化并启动 SIP 引擎
    int start(const char *server_ip, int server_port, const char *local_ip, int local_port, const char *device_id, const char *password);

    // 停止并退出
    void stop();

private:
    // 注册函数
    void register_to_server();

    // 心跳线程函数
    void heartbeat_loop();

    // 信令处理循环（这是核心，处理平台发来的 Invite 等消息）
    void event_loop();

private:
    struct eXosip_t *ctx;      // eXosip 上下文（整个 SIP 引擎）
    bool is_running;           // 控制事件线程退出
    int register_id;           // 注册 ID
    std::thread *event_thread; // SIP 事件监听线程
    std::string device_id_;    // 保存设备 ID，后续生成 XML 会用
};

#endif