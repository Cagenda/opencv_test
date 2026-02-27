#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

class MediaStreamer
{
public:
    MediaStreamer();
    ~MediaStreamer();

    int start(const std::string& ip, int port, const std::string& ssrc);
    void stop();
    bool is_running() const;

private:
    void run_loop();

    std::atomic<bool> running_{false};
    std::thread worker_;
    std::string ip_;
    int port_{0};
    std::string ssrc_;
    uint32_t ssrc_u32_{0};
};
