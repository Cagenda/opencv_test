#include "gb28181_agent.h"
#include <iostream>
#include <unistd.h>

// 初始化参数列表，将ctx=nullptr,is_running=false, heartbeat_thread=nullptr
GB28181Agent::GB28181Agent()
    : ctx(nullptr), is_running(false), heartbeat_thread(nullptr), call_id(-1),
      dialog_id(-1), is_pushing(false) {}
// 调用stop()函数
GB28181Agent::~GB28181Agent() { stop(); }

int GB28181Agent::start(const char *server_ip, int server_port,
                        const char *local_ip, int local_port,
                        const char *device_id, const char *password)
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

    // ================= [修改] 保存参数到成员变量 =================
    device_id_ = device_id;
    server_ip_ = server_ip;
    server_port_ = server_port;
    local_ip_ = local_ip;
    local_port_ = local_port;

    is_running = true;

    // 3. ==============构建注册消息====================
    osip_message_t *reg_msg =
        nullptr; // 保存“即将发送出去的 REGISTER SIP 报文对象
    char from_sip[100], proxy_sip[100];

    // 格式：sip:设备ID@服务器IP:端口
    // from_sip 本质上是一个 SIP URI，表示“本设备的 SIP 身份”
    sprintf(from_sip, "sip:%s@%s:%d", device_id, server_ip, server_port);

    // proxy_sip 也是一个 SIP URI，表示“要把 REGISTER 发到哪里”（也就是 SIP
    // 服务器/代理/注册服务器地址
    sprintf(proxy_sip, "sip:%s:%d", server_ip, server_port);

    eXosip_lock(ctx);
    // ===========4.关键：构建 REGISTER 请求=============
    // 1.ctx 内部创建“注册会话对象”，并返回它的编号 register_id（类似于ctx句柄）
    // 2.构造出一条“初始 REGISTER 请求报文”，并写到 reg_msg
    register_id = eXosip_register_build_initial_register(ctx, from_sip, proxy_sip,
                                                         nullptr, 3600, &reg_msg);
    if (register_id < 0)
    {
        std::cerr << "Build register failed!" << std::endl;
        eXosip_unlock(ctx);
        return -1;
    }

    // ===============5.添加鉴权信息=====================
    // 这一步并不会立刻让你发的第一条 REGISTER 带上 Authorization。
    eXosip_add_authentication_info(ctx, device_id, device_id, password, "MD5",
                                   nullptr);

    // ==============6.发送Register====================
    // 第一次发通常不带 Authorization
    eXosip_register_send_register(ctx, register_id, reg_msg);
    eXosip_unlock(ctx);
    std::cout << "SIP Register sent to " << server_ip << std::endl;

    // =============5. 启动事件监听线程==================
    event_thread = new std::thread(&GB28181Agent::event_loop, this);

    return 0;
}

// gb28181_agent.cpp

void GB28181Agent::heartbeat_loop()
{
    int sn = 1; // 心跳序列号
    while (is_running)
    {
        // 1. 构建心跳 XML (保持不变)
        std::stringstream ss;
        ss << "<?xml version=\"1.0\" encoding=\"GB2312\"?>\r\n";
        ss << "<Notify>\r\n";
        ss << "<CmdType>Keepalive</CmdType>\r\n";
        ss << "<SN>" << sn++ << "</SN>\r\n";
        ss << "<DeviceID>" << device_id_ << "</DeviceID>\r\n";
        ss << "<Status>OK</Status>\r\n";
        ss << "</Notify>\r\n";

        std::string body = ss.str();

        // 2. 动态构建 SIP 地址
        // 目标格式: sip:服务器ID@服务器IP:端口
        // (服务器ID通常就是域ID，这里暂用设备ID前10位或直接发给服务器IP)
        // 为了通用，我们直接发给 "sip:server_ip:server_port"

        char server_uri[100];
        char local_uri[100];

        // 1. IP 和 Port：负责“物理传输” (Socket层)
        //    告诉 eXosip 底层：把这个包通过网线发到 192.168.137.1 的 5060 端口。
        //    如果没有这两个，数据包连网卡都出不去。
        sprintf(server_uri, "sip:%s:%d", server_ip_.c_str(), server_port_);

        // 构建本地 URI (比如 sip:3402000...001@192.168.137.1:5060)
        // 注意：From 头域通常需要包含设备ID
        sprintf(local_uri, "sip:%s@%s:%d", device_id_.c_str(), server_ip_.c_str(),
                server_port_);

        osip_message_t *request = nullptr;

        eXosip_lock(ctx);
        // 使用动态构建的 URI 发送 MESSAGE
        int ret = eXosip_message_build_request(ctx, &request, "MESSAGE",
                                               server_uri, // To: 服务器
                                               local_uri,  // From: 设备自己
                                               nullptr);

        if (ret == 0 && request != nullptr)
        {
            osip_message_set_body(request, body.c_str(), body.length());
            osip_message_set_content_type(request, "Application/MANSCDP+xml");
            eXosip_message_send_request(ctx, request);
            std::cout << ">>> 发送心跳 SN=" << (sn - 1) << " To: " << server_uri
                      << std::endl;
        }
        eXosip_unlock(ctx);

        // 3. 休眠 60 秒
        sleep(60);
    }
}

void GB28181Agent::event_loop()
{
    // 只要系统处于运行状态，就持续循环处理信令
    while (is_running)
    {
        // 1. 等待 SIP 事件。第二个参数 0 表示不处理外部 socket，第三个参数 50
        // 表示最多阻塞等待 50 毫秒 如果 50ms 内没有任何信令消息，evt 将为 nullptr
        eXosip_event_t *evt = eXosip_event_wait(ctx, 0, 50);

        // 2. 驱动 eXosip 内部的维护逻辑（如：自动响应 401
        // 鉴权挑战、自动处理超时重发等）
        eXosip_lock(ctx);
        eXosip_automatic_action(ctx);
        eXosip_unlock(ctx);

        // 如果没有事件发生，继续下一次循环
        if (!evt)
            continue;

        // 3. 根据事件类型进行分类处理
        if (evt->type == EXOSIP_REGISTRATION_SUCCESS)
        {
            // --- 处理注册成功 ---
            std::cout << ">>> [事件] 注册成功 (200 OK) <<<" << std::endl;
            // 只有当注册成功后，我们才启动心跳线程，开始定时向服务器发送心跳包
            if (heartbeat_thread == nullptr)
            {
                heartbeat_thread = new std::thread(&GB28181Agent::heartbeat_loop, this);
            }
        }
        else if (evt->type == EXOSIP_REGISTRATION_FAILURE)
        {
            // --- 处理注册失败 ---
            std::cout << ">>> [警告] 注册失败 (可能是密码错误或网络连接超时) <<<"
                      << std::endl;
        }
        else if (evt->type == EXOSIP_CALL_INVITE)
        {
            // =================================================================================
            // 核心业务：处理平台点播请求 (INVITE)
            // 流程：收到 INVITE -> 提取头部 -> 解析 SDP (对方收流IP/端口) -> 回复 200
            // OK (带本机 SDP)
            // =================================================================================
            std::cout << "\n>>> [信令] 收到平台 INVITE 请求 <<<" << std::endl;

            // [调试技巧] 打印核心 ID，用于 Wireshark 过滤和日志对齐
            // cid (Call-ID): 标识一次完整的通话会话
            // did (Dialog-ID): eXosip 内部的会话句柄
            // tid (Transaction-ID): 标识当前这次请求/响应的事务
            std::cout << "  [Event IDs] cid=" << evt->cid << ", did=" << evt->did
                      << ", tid=" << evt->tid << std::endl;

            // 获取 INVITE 请求消息对象
            osip_message_t *invite = evt->request;

            // --- 1. 提取 SIP 头部信息 ---
            // 作用：将 SIP 头从结构体转换为字符串，便于调试日志输出
            // Call-ID: 唯一标识一次通话会话，用于关联 INVITE/ACK/BYE
            // From: 主叫方（平台），包含平台的 SIP URI
            // To: 被叫方（设备），包含设备的 SIP URI 和 tag 参数
            char *call_id_str = nullptr;
            char *from_str = nullptr;
            char *to_str = nullptr;
            osip_call_id_to_str(invite->call_id, &call_id_str);
            osip_from_to_str(invite->from, &from_str);
            osip_to_to_str(invite->to, &to_str);

            // 打印 SIP 头部信息到日志
            // 这些信息在多路并发点播和故障排查时非常重要
            std::cout << "  [SIP Headers]" << std::endl;
            if (call_id_str)
                std::cout << "    Call-ID: " << call_id_str << std::endl;
            if (invite->cseq)
                std::cout << "    CSeq: " << invite->cseq->number << " "
                          << invite->cseq->method << std::endl;
            if (from_str)
                std::cout << "    From: " << from_str << std::endl;
            if (to_str)
                std::cout << "    To: " << to_str << std::endl;

            // 释放字符串内存（osip_xxx_to_str 函数内部使用 malloc 分配）
            osip_free(call_id_str);
            osip_free(from_str);
            osip_free(to_str);

            // --- 2. 获取对方的 SDP (Session Description Protocol) ---
            // 多线程安全：eXosip 库的 API 必须在 eXosip_lock/unlock 保护下调用
            // SDP 描述了媒体会话信息，包含平台的收流 IP、端口等参数
            eXosip_lock(ctx);
            sdp_message_t *remote_sdp = eXosip_get_remote_sdp(ctx, evt->did);
            eXosip_unlock(ctx);

            // 调试输出：检查 SDP 是否成功获取
            std::cout << "  [调试] eXosip_get_remote_sdp 返回: "
                      << (remote_sdp ? "成功" : "NULL") << std::endl;

            // --- 3. 从 SDP 中解析推流目标地址 ---
            // 如果 eXosip 成功解析了 SDP，则从结构体中提取信息
            if (remote_sdp)
            {
                // 解析 c= 行：Connection Data (连接数据)
                // 格式：c=IN IP4 192.168.1.100
                // 这个 IP 是平台流媒体服务器的地址，设备需要往这个 IP 推流
                if (remote_sdp->c_connection)
                    push_ip = remote_sdp->c_connection->c_addr;

                // 解析 m= 行：Media Description (媒体描述)
                // 格式：m=video 6000 RTP/AVP 96
                // 6000 是平台流媒体服务器的 RTP 接收端口
                if (!osip_list_eol(&remote_sdp->m_medias, 0))
                {
                    sdp_media_t *media =
                        (sdp_media_t *)osip_list_get(&remote_sdp->m_medias, 0);
                    push_port = atoi(media->m_port);
                }
            }

            // --- 4. 从原始 SDP 文本中解析 GB28181 特有字段 ---
            // eXosip 的标准 SDP 解析器不识别 GB28181 的 y= 字段（SSRC）
            // 因此需要从原始文本（Body）中手动提取
            // 同时，这也是 remote_sdp 失败时的备用解析方案
            osip_body_t *sdp_body_raw = nullptr;
            osip_message_get_body(invite, 0, &sdp_body_raw);
            if (sdp_body_raw && sdp_body_raw->body)
            {
                std::string sdp_str(sdp_body_raw->body);

                // 备用解析：当 eXosip_get_remote_sdp 返回 NULL 时的兜底方案
                // 使用字符串查找直接从 SDP 文本中提取 IP 和端口
                if (!remote_sdp)
                {
                    std::cout << "  [备用解析] 从原始 SDP Body 中手动提取参数..."
                              << std::endl;

                    // 查找 c= 行并提取 IP 地址
                    // 示例：c=IN IP4 192.168.1.100
                    //           ^^^^^^^^^^^^^^^^ (从第9个字符开始提取)
                    size_t c_pos = sdp_str.find("c=IN IP4 ");
                    if (c_pos != std::string::npos)
                    {
                        size_t ip_start = c_pos + 9; // 跳过 "c=IN IP4 "
                        size_t ip_end = sdp_str.find("\r\n", ip_start);
                        if (ip_end == std::string::npos)
                            ip_end = sdp_str.find("\n", ip_start); // 兼容不同换行符
                        if (ip_end != std::string::npos)
                            push_ip = sdp_str.substr(ip_start, ip_end - ip_start);
                    }

                    // 查找 m= 行并提取端口号
                    // 示例：m=video 6000 RTP/AVP 96
                    //               ^^^^ (提取这个端口)
                    size_t m_pos = sdp_str.find("m=video ");
                    if (m_pos != std::string::npos)
                    {
                        size_t port_start = m_pos + 8;                   // 跳过 "m=video "
                        size_t port_end = sdp_str.find(" ", port_start); // 找到空格
                        if (port_end != std::string::npos)
                            push_port = atoi(
                                sdp_str.substr(port_start, port_end - port_start).c_str());
                    }
                }

                // 解析 GB28181 特有的 y= 字段（SSRC - 同步源标识符）
                // 示例：y=0100001234
                // SSRC 用于 RTP 流的唯一标识，设备必须在推流时使用同一个 SSRC
                // 并在 200 OK 响应的 SDP 中原样回传给平台
                size_t y_pos = sdp_str.find("y=");
                if (y_pos != std::string::npos)
                {
                    size_t end_pos = sdp_str.find("\r\n", y_pos);
                    if (end_pos == std::string::npos)
                        end_pos = sdp_str.length();
                    push_ssrc = sdp_str.substr(y_pos + 2, end_pos - (y_pos + 2));
                }
            }

            // 打印解析结果：推流目标信息
            std::cout << "  [SDP解析] 推流目标 IP: " << push_ip
                      << ", 推流目标端口: " << push_port << ", SSRC: " << push_ssrc
                      << std::endl;

            // --- 5. 构建 200 OK 响应 ---
            // 必须在收到 INVITE 后立即回复 200 OK，告诉平台"我接受这次通话"
            osip_message_t *answer = nullptr;
            eXosip_lock(ctx); // 加锁保护 eXosip 内部状态
            // 为当前事务 (tid) 构建一个 200 状态码的应答消息
            int build_ret = eXosip_call_build_answer(ctx, evt->tid, 200, &answer);

            if (build_ret != 0 || !answer)
            {
                // 构建失败，打印错误日志
                std::cerr << "!!! [错误] eXosip_call_build_answer 失败! ret="
                          << build_ret << " !!!" << std::endl;
            }
            else
            {
                // --- 6. 填充设备侧的 SDP 到 200 OK 响应中 ---
                // SDP 告诉平台：我这边的媒体参数（IP、端口、编码格式等）
                std::string sdp_body = "v=0\r\n"; // SDP 版本号

                // o= 行：所有者/创建者 (Originator)
                // 格式：o=<username> <sess-id> <sess-version> <nettype> <addrtype>
                // <unicast-address> 关键：必须填本机 IP
                // (local_ip_)，告诉平台"流从我这里发出"
                sdp_body += "o=" + device_id_ + " 0 0 IN IP4 " + local_ip_ + "\r\n";

                sdp_body += "s=Play\r\n"; // s= 行：会话名称

                // c= 行：连接信息 (Connection Data)
                // 关键：同样必须填本机 IP，如果填成平台 IP 会导致 "SMS not found" 错误
                sdp_body += "c=IN IP4 " + local_ip_ + "\r\n";

                sdp_body += "t=0 0\r\n"; // t= 行：时间描述（0 0 表示永久会话）

                // m= 行：媒体描述 (Media Description)
                // 格式：m=<media> <port> <proto> <fmt>
                // 9000 是占位端口，在 sendonly 模式下这个值不重要
                // 实际推流目标是上面解析出的 push_ip:push_port
                sdp_body += "m=video 9000 RTP/AVP 96\r\n";

                sdp_body += "a=sendonly\r\n";           // a= 行：属性（sendonly 表示只推流）
                sdp_body += "a=rtpmap:96 PS/90000\r\n"; // RTP 负载类型映射（PS 是
                                                        // GB28181 的标准封装）

                // y= 行：SSRC（必须原样回传平台发来的 SSRC）
                sdp_body += "y=" + push_ssrc + "\r\n";

                // 打印构造的 SDP 内容（用于调试）
                std::cout << "  [调试] 构造的 SDP:\n"
                          << sdp_body << std::endl;

                // 将 SDP 文本设置到 200 OK 消息的 Body 中
                osip_message_set_body(answer, sdp_body.c_str(), sdp_body.length());
                // 设置 Content-Type 头域为 "application/sdp"
                osip_message_set_content_type(answer, "application/sdp");

                // --- 7. 发送 200 OK 响应给平台 ---
                int ret = eXosip_call_send_answer(ctx, evt->tid, 200, answer);
                if (ret == 0)
                {
                    std::cout << ">>> [信令] 回复 200 OK (携带 SDP) 成功 <<<"
                              << std::endl;
                }
                else
                {
                    std::cerr << ">>> [错误] 回复 200 OK 失败! ret=" << ret << " <<<"
                              << std::endl;
                }

                // 保存会话标识符，用于后续的 BYE (挂断) 请求
                call_id = evt->cid;
                dialog_id = evt->did;
            }
            eXosip_unlock(ctx);
        }
        else if (evt->type == EXOSIP_CALL_ACK)
        {

            // --- 处理 ACK 确认 ---
            // 收到这个事件意味着三次握手完成，推流链路正式打通
            std::cout << ">>> [事件] 收到平台 ACK (cid=" << evt->cid
                      << "), 三次握手彻底完成！可以推流项目了 <<<" << std::endl;
            is_pushing = true;

            // 【关键命令】立刻动态启动一个推流工作推手！
            streamer.start(push_ip, push_port, push_ssrc);
        }
        else if (evt->type == EXOSIP_CALL_CLOSED)
        {
            // --- 处理挂断消息 (BYE) ---
            std::cout << ">>> [事件] 收到挂断请求 (BYE/CLOSED)，停止传输 <<<"
                      << std::endl;
            is_pushing = false;
            streamer.stop();
            call_id = -1;
            dialog_id = -1;
        }

        // --- 处理 SIP MESSAGE (如 Catalog 查询、云台控制等) ---
        else if (evt->type == EXOSIP_MESSAGE_NEW)
        {

            osip_message_t *msg = evt->request;
            osip_body_t *body = nullptr;
            osip_message_get_body(msg, 0, &body);

            if (body && body->body)
            {
                std::string body_str(body->body);

                // 【关键触发点】：你的代码在这里检查，平台发来的 XML 里有没有 "Catalog" 这个词？
                // 如果有，说明平台正在向你发送“目录查询命令”（要看你的菜单）。
                // 检查是否是 Catalog (目录查询) 命令
                if (body_str.find("CmdType") != std::string::npos &&
                    body_str.find("Catalog") != std::string::npos)
                {
                    std::cout << ">>> [事件] 收到 Catalog 查询请求 <<<" << std::endl;

                    // 提取 SN (序列号)
                    std::string sn = "1";
                    size_t sn_pos = body_str.find("<SN>");
                    if (sn_pos != std::string::npos)
                    {
                        size_t sn_end = body_str.find("</SN>", sn_pos);
                        if (sn_end != std::string::npos)
                        {
                            sn = body_str.substr(sn_pos + 4, sn_end - (sn_pos + 4));
                        }
                    }

                    // 1. 先快速回复 200 OK 表示消息已收到
                    osip_message_t *answer = nullptr;
                    eXosip_lock(ctx);
                    eXosip_message_build_answer(ctx, evt->tid, 200, &answer);
                    eXosip_message_send_answer(ctx, evt->tid, 200, answer);
                    eXosip_unlock(ctx);

                    // 2. 发送实际的 Catalog 资源列表 XML (异步响应)
                    char to_uri[100];
                    sprintf(to_uri, "sip:%s:%d", server_ip_.c_str(), server_port_);
                    send_catalog(to_uri, sn.c_str());
                }
            }
        }

        // 4. 重要：每处理完一个事件，必须调用此函数释放内存，防止内存泄漏
        eXosip_event_free(evt);
    }
}

void GB28181Agent::stop()
{
    is_running = false;
    if (event_thread && event_thread->joinable())
    {
        event_thread->join();
        delete event_thread;
        event_thread = nullptr;
    }

    if (heartbeat_thread && heartbeat_thread->joinable())
    {
        heartbeat_thread->join();
        delete heartbeat_thread;
        heartbeat_thread = nullptr;
    }
    eXosip_quit(ctx);
    osip_free(ctx);
}

void GB28181Agent::send_catalog(const char *to_sip_uri, const char *sn)
{
    // 构建 Catalog 响应 XML
    std::stringstream ss;
    // 国标规定 XML 通常使用 GB2312 编码
    ss << "<?xml version=\"1.0\" encoding=\"GB2312\"?>\r\n";
    ss << "<Response>\r\n";
    ss << "<CmdType>Catalog</CmdType>\r\n";
    ss << "<SN>" << (sn ? sn : "1") << "</SN>\r\n";        // 使用请求中的序列号
    ss << "<DeviceID>" << device_id_ << "</DeviceID>\r\n"; // 根节点为设备 ID
    ss << "<SumNum>1</SumNum>\r\n";                        // 通道总数：1
    ss << "<DeviceList Num=\"1\">\r\n";
    ss << "<Item>\r\n";
    // 通道 ID（DeviceID）：通常为 20 位。
    ss << "<DeviceID>" << device_id_ << "</DeviceID>\r\n";
    ss << "<Name>RK3588_Camera_01</Name>\r\n";
    ss << "<Manufacturer>OpenCV_GB28181</Manufacturer>\r\n";
    ss << "<Model>RK3588</Model>\r\n";
    ss << "<Owner>Owner</Owner>\r\n";
    ss << "<CivilCode>CivilCode</CivilCode>\r\n";
    ss << "<Address>Address</Address>\r\n";
    // ParentID 字段非常有重要：它建立了通道与所属设备之间的层级关系。
    ss << "<ParentID>" << device_id_ << "</ParentID>\r\n";
    ss << "<Parental>0</Parental>\r\n";
    ss << "<SafetyWay>0</SafetyWay>\r\n";
    ss << "<RegisterWay>1</RegisterWay>\r\n";
    ss << "<Secrecy>0</Secrecy>\r\n";
    ss << "<Status>ON</Status>\r\n"; // 在线状态
    ss << "</Item>\r\n";
    ss << "</DeviceList>\r\n";
    ss << "</Response>\r\n";

    std::string body = ss.str();

    // 构建并发送 SIP MESSAGE 信令
    osip_message_t *request = nullptr;
    char local_uri[100];
    // From 字段：通常格式为 sip:设备ID@本设备IP:端口
    sprintf(local_uri, "sip:%s@%s:%d", device_id_.c_str(), server_ip_.c_str(),
            server_port_);

    eXosip_lock(ctx);
    // 构建 MESSAGE 请求
    int ret = eXosip_message_build_request(ctx, &request, "MESSAGE", to_sip_uri,
                                           local_uri, nullptr);
    if (ret == 0 && request != nullptr)
    {
        osip_message_set_body(request, body.c_str(), body.length());
        osip_message_set_content_type(request, "Application/MANSCDP+xml");
        eXosip_message_send_request(ctx, request);
        std::cout << ">>> [目录上报] 已发送 Catalog 响应 XML (包含通道: "
                  << device_id_ << ") <<<" << std::endl;
    }
    else
    {
        std::cerr << "!!! [错误] 构建 Catalog 响应失败: " << ret << " !!!"
                  << std::endl;
    }
    eXosip_unlock(ctx);
}