#include "gb28181_agent.h"
#include <iostream>
#include <unistd.h>

// 初始化参数列表，将ctx=nullptr,is_running=false
GB28181Agent::GB28181Agent() : ctx(nullptr), is_running(false) {}
// 调用stop()函数
GB28181Agent::~GB28181Agent() { stop(); }

int GB28181Agent::start(const char *server_ip, int server_port, const char *local_ip, int local_port, const char *device_id, const char *password)
{
    // 1. ===================初始化 eXosip =============
    ctx = eXosip_malloc();
    if (eXosip_init(ctx))
    {
        std::cerr << "eXosip init failed!" << std::endl;
        return -1;
    }

    // 2. ======监听本地 SIP 端口通常是 5060）==========
    // 你至少要知道：单网卡就用 nullptr，多网卡/指定出口就用具体 IP
    if (eXosip_listen_addr(ctx, IPPROTO_UDP, nullptr, local_port, AF_INET, 0))
    {
        std::cerr << "eXosip listen failed!" << std::endl;
        return -1;
    }

    device_id_ = device_id;
    is_running = true;

    // 3. ==============构建注册消息====================
    osip_message_t *reg_msg = nullptr; // 保存“即将发送出去的 REGISTER SIP 报文对象
    char from_sip[100], proxy_sip[100];

    // 格式：sip:设备ID@服务器IP:端口
    // from_sip 本质上是一个 SIP URI，表示“本设备的 SIP 身份”
    sprintf(from_sip, "sip:%s@%s:%d", device_id, server_ip, server_port);

    // proxy_sip 也是一个 SIP URI，表示“要把 REGISTER 发到哪里”（也就是 SIP 服务器/代理/注册服务器地址
    sprintf(proxy_sip, "sip:%s:%d", server_ip, server_port);

    eXosip_lock(ctx);
    // ===========4.关键：构建 REGISTER 请求=============
    // 1.ctx 内部创建“注册会话对象”，并返回它的编号 register_id（类似于ctx句柄）
    // 2.构造出一条“初始 REGISTER 请求报文”，并写到 reg_msg
    register_id = eXosip_register_build_initial_register(ctx, from_sip, proxy_sip, nullptr, 3600, &reg_msg);
    if (register_id < 0)
    {
        std::cerr << "Build register failed!" << std::endl;
        eXosip_unlock(ctx);
        return -1;
    }

    // ===============5.添加鉴权信息=====================
    // 这一步并不会立刻让你发的第一条 REGISTER 带上 Authorization。
    eXosip_add_authentication_info(ctx, device_id, device_id, password, "MD5", nullptr);

    // ==============6.发送Register====================
    // 第一次发通常不带 Authorization
    eXosip_register_send_register(ctx, register_id, reg_msg);
    eXosip_unlock(ctx);
    std::cout << "SIP Register sent to " << server_ip << std::endl;

    // =============5. 启动事件监听线程==================
    event_thread = new std::thread(&GB28181Agent::event_loop, this);

    return 0;
}

void GB28181Agent::event_loop()
{
    while (is_running)
    {
        // 去 ctx（SIP 引擎实例）的事件队列里取一个事件最多等 50ms：50ms 内有事件 → 返回 evt。没有事件 → 返回 nullptr
        eXosip_event_t *evt = eXosip_event_wait(ctx, 0, 50); // 等待 50ms
        eXosip_lock(ctx);
        eXosip_automatic_action(ctx); // 这句很关键，它会驱动 eXosip 内部的“自动处理逻辑”，最常见的是：遇到 401/407（Digest 认证挑战）时自动重发 REGISTER
        eXosip_unlock(ctx);

        if (!evt) // 表示evt为nullptr，为空事件
            continue;

        if (evt->type == EXOSIP_REGISTRATION_SUCCESS)
        {
            std::cout << ">>> 注册成功 (200 OK) <<<" << std::endl;
            // 可以在这里启动心跳线程
        }
        else if (evt->type == EXOSIP_REGISTRATION_FAILURE)
        {
            std::cout << ">>> 注册失败 (可能是密码错误或网络不通) <<<" << std::endl;
        }

        eXosip_event_free(evt);
    }
}

void GB28181Agent::stop()
{
    is_running = false;
    if (event_thread && event_thread->joinable())
        event_thread->join();
    eXosip_quit(ctx);
    osip_free(ctx);
}