#include "media_streamer.h"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <opencv2/imgproc.hpp>
#include <shared_mutex>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

extern cv::Mat g_mosaic_canvas;
extern std::shared_mutex g_mosaic_mutex;

static uint32_t parse_ssrc_u32(const std::string &s)
{
    try
    {
        unsigned long v = std::stoul(s, nullptr, 10);
        return static_cast<uint32_t>(v);
    }
    catch (...)
    {
        return static_cast<uint32_t>(std::hash<std::string>{}(s));
    }
}

static void append_u16(std::vector<uint8_t> &out, uint16_t v)
{
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 0) & 0xFF));
}

static void append_u32(std::vector<uint8_t> &out, uint32_t v)
{
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 0) & 0xFF));
}

static uint32_t mpeg_crc32(const uint8_t *data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
    {
        crc ^= (static_cast<uint32_t>(data[i]) << 24);
        for (int b = 0; b < 8; b++)
        {
            if (crc & 0x80000000u)
                crc = (crc << 1) ^ 0x04C11DB7u;
            else
                crc <<= 1;
        }
    }
    return crc;
}

static std::vector<uint8_t> build_ps_pack_header()
{
    std::vector<uint8_t> out;
    static const uint8_t base[14] = {
        0x00, 0x00, 0x01, 0xBA,
        0x44, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x02, 0x5F, 0x03,
        0xFE};
    out.insert(out.end(), base, base + sizeof(base));
    for (int i = 0; i < 6; i++)
        out.push_back(0xFF);
    return out;
}

static std::vector<uint8_t> build_psm()
{
    std::vector<uint8_t> psm;
    psm.push_back(0x00);
    psm.push_back(0x00);
    psm.push_back(0x01);
    psm.push_back(0xBC);

    std::vector<uint8_t> body;
    body.push_back(0xE0);
    body.push_back(0x00);
    body.push_back(0x00);

    std::vector<uint8_t> es_map;
    es_map.push_back(0x1B);
    es_map.push_back(0xE0);
    es_map.push_back(0x00);
    es_map.push_back(0x00);

    uint16_t program_stream_info_length = 0;
    uint16_t elementary_stream_map_length = static_cast<uint16_t>(es_map.size());

    append_u16(body, program_stream_info_length);
    append_u16(body, elementary_stream_map_length);
    body.insert(body.end(), es_map.begin(), es_map.end());

    uint32_t crc = mpeg_crc32(body.data(), body.size());
    append_u32(body, crc);

    append_u16(psm, static_cast<uint16_t>(body.size()));
    psm.insert(psm.end(), body.begin(), body.end());
    return psm;
}

static void append_pts(std::vector<uint8_t> &out, uint64_t pts90k)
{
    uint64_t pts = pts90k & 0x1FFFFFFFFull;
    uint8_t b0 = static_cast<uint8_t>((0x2 << 4) | (((pts >> 30) & 0x7) << 1) | 0x1);
    uint8_t b1 = static_cast<uint8_t>((pts >> 22) & 0xFF);
    uint8_t b2 = static_cast<uint8_t>((((pts >> 15) & 0x7F) << 1) | 0x1);
    uint8_t b3 = static_cast<uint8_t>((pts >> 7) & 0xFF);
    uint8_t b4 = static_cast<uint8_t>((((pts >> 0) & 0x7F) << 1) | 0x1);
    out.push_back(b0);
    out.push_back(b1);
    out.push_back(b2);
    out.push_back(b3);
    out.push_back(b4);
}

static std::vector<uint8_t> build_pes_video(uint64_t pts90k, const uint8_t *payload, size_t payload_len)
{
    std::vector<uint8_t> out;
    out.push_back(0x00);
    out.push_back(0x00);
    out.push_back(0x01);
    out.push_back(0xE0);
    out.push_back(0x00);
    out.push_back(0x00);
    out.push_back(0x80);
    out.push_back(0x80);
    out.push_back(0x05);
    append_pts(out, pts90k);
    out.insert(out.end(), payload, payload + payload_len);
    return out;
}

static void rtp_send_ps(int sockfd,
                        const sockaddr_in &dst,
                        uint16_t &seq,
                        uint32_t ssrc,
                        uint32_t timestamp,
                        const uint8_t *ps,
                        size_t ps_len)
{
    const size_t max_payload = 1400;
    size_t offset = 0;
    while (offset < ps_len)
    {
        size_t chunk = ps_len - offset;
        if (chunk > max_payload)
            chunk = max_payload;

        bool marker = (offset + chunk) >= ps_len;

        uint8_t rtp[12];
        rtp[0] = 0x80;
        rtp[1] = static_cast<uint8_t>(96 & 0x7F);
        if (marker)
            rtp[1] |= 0x80;
        rtp[2] = static_cast<uint8_t>((seq >> 8) & 0xFF);
        rtp[3] = static_cast<uint8_t>((seq) & 0xFF);
        rtp[4] = static_cast<uint8_t>((timestamp >> 24) & 0xFF);
        rtp[5] = static_cast<uint8_t>((timestamp >> 16) & 0xFF);
        rtp[6] = static_cast<uint8_t>((timestamp >> 8) & 0xFF);
        rtp[7] = static_cast<uint8_t>((timestamp) & 0xFF);
        rtp[8] = static_cast<uint8_t>((ssrc >> 24) & 0xFF);
        rtp[9] = static_cast<uint8_t>((ssrc >> 16) & 0xFF);
        rtp[10] = static_cast<uint8_t>((ssrc >> 8) & 0xFF);
        rtp[11] = static_cast<uint8_t>((ssrc) & 0xFF);

        std::vector<uint8_t> pkt;
        pkt.insert(pkt.end(), rtp, rtp + sizeof(rtp));
        pkt.insert(pkt.end(), ps + offset, ps + offset + chunk);

        sendto(sockfd, pkt.data(), pkt.size(), 0, reinterpret_cast<const sockaddr *>(&dst), sizeof(dst));

        seq++;
        offset += chunk;
    }
}

MediaStreamer::MediaStreamer() = default;

MediaStreamer::~MediaStreamer()
{
    stop();
}

int MediaStreamer::start(const std::string &ip, int port, const std::string &ssrc)
{
    stop();
    ip_ = ip;
    port_ = port;
    ssrc_ = ssrc;
    ssrc_u32_ = parse_ssrc_u32(ssrc_);
    running_.store(true);
    worker_ = std::thread(&MediaStreamer::run_loop, this);
    return 0;
}

void MediaStreamer::stop()
{
    running_.store(false);
    if (worker_.joinable())
        worker_.join();
}

bool MediaStreamer::is_running() const
{
    return running_.load();
}

void MediaStreamer::run_loop()
{
    const int fps = 25;
    const int64_t frame_interval_us = 1000000 / fps;

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
    {
        running_.store(false);
        return;
    }

    sockaddr_in dst;
    std::memset(&dst, 0, sizeof(dst));
    dst.sin_family = AF_INET;
    dst.sin_port = htons(static_cast<uint16_t>(port_));
    if (inet_pton(AF_INET, ip_.c_str(), &dst.sin_addr) != 1)
    {
        close(sockfd);
        running_.store(false);
        return;
    }

    const AVCodec *codec = avcodec_find_encoder_by_name("h264_v4l2m2m");
    if (!codec)
        codec = avcodec_find_encoder_by_name("libx264");
    if (!codec)
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec)
    {
        close(sockfd);
        running_.store(false);
        return;
    }

    int width = 0;
    int height = 0;
    while (running_.load())
    {
        {
            std::shared_lock<std::shared_mutex> lock(g_mosaic_mutex);
            if (!g_mosaic_canvas.empty())
            {
                width = g_mosaic_canvas.cols;
                height = g_mosaic_canvas.rows;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!running_.load() || width <= 0 || height <= 0)
    {
        close(sockfd);
        return;
    }

    AVCodecContext *enc = avcodec_alloc_context3(codec);
    if (!enc)
    {
        close(sockfd);
        running_.store(false);
        return;
    }

    enc->width = width;
    enc->height = height;
    enc->pix_fmt = AV_PIX_FMT_YUV420P;
    enc->time_base = AVRational{1, fps};
    enc->framerate = AVRational{fps, 1};
    enc->gop_size = fps * 2;
    enc->max_b_frames = 0;
    enc->bit_rate = 2'000'000;

    AVDictionary *opts = nullptr;
    if (codec->name && std::string(codec->name) == "libx264")
    {
        av_dict_set(&opts, "preset", "veryfast", 0);
        av_dict_set(&opts, "tune", "zerolatency", 0);
        av_dict_set(&opts, "profile", "baseline", 0);
    }

    if (avcodec_open2(enc, codec, &opts) < 0)
    {
        av_dict_free(&opts);
        avcodec_free_context(&enc);
        close(sockfd);
        running_.store(false);
        return;
    }
    av_dict_free(&opts);

    const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
    AVBSFContext *bsf_ctx = nullptr;
    if (bsf)
    {
        if (av_bsf_alloc(bsf, &bsf_ctx) == 0)
        {
            avcodec_parameters_from_context(bsf_ctx->par_in, enc);
            if (av_bsf_init(bsf_ctx) != 0)
            {
                av_bsf_free(&bsf_ctx);
                bsf_ctx = nullptr;
            }
        }
    }

    SwsContext *sws = sws_getContext(width, height, AV_PIX_FMT_BGR24,
                                     width, height, AV_PIX_FMT_YUV420P,
                                     SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws)
    {
        if (bsf_ctx)
            av_bsf_free(&bsf_ctx);
        avcodec_free_context(&enc);
        close(sockfd);
        running_.store(false);
        return;
    }

    AVFrame *frame = av_frame_alloc();
    if (!frame)
    {
        sws_freeContext(sws);
        if (bsf_ctx)
            av_bsf_free(&bsf_ctx);
        avcodec_free_context(&enc);
        close(sockfd);
        running_.store(false);
        return;
    }

    frame->format = enc->pix_fmt;
    frame->width = width;
    frame->height = height;
    if (av_frame_get_buffer(frame, 32) < 0)
    {
        av_frame_free(&frame);
        sws_freeContext(sws);
        if (bsf_ctx)
            av_bsf_free(&bsf_ctx);
        avcodec_free_context(&enc);
        close(sockfd);
        running_.store(false);
        return;
    }

    AVPacket *pkt = av_packet_alloc();
    AVPacket *out_pkt = av_packet_alloc();
    if (!pkt || !out_pkt)
    {
        if (pkt)
            av_packet_free(&pkt);
        if (out_pkt)
            av_packet_free(&out_pkt);
        av_frame_free(&frame);
        sws_freeContext(sws);
        if (bsf_ctx)
            av_bsf_free(&bsf_ctx);
        avcodec_free_context(&enc);
        close(sockfd);
        running_.store(false);
        return;
    }

    uint16_t seq = 0;
    uint64_t frame_index = 0;
    bool sent_psm = false;
    auto next_tick = std::chrono::steady_clock::now();

    while (running_.load())
    {
        cv::Mat bgr;
        {
            std::shared_lock<std::shared_mutex> lock(g_mosaic_mutex);
            if (!g_mosaic_canvas.empty())
                bgr = g_mosaic_canvas.clone();
        }
        if (bgr.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        if (bgr.cols != width || bgr.rows != height || bgr.type() != CV_8UC3)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        if (av_frame_make_writable(frame) < 0)
            break;

        const uint8_t *in_data[4] = {bgr.data, nullptr, nullptr, nullptr};
        int in_linesize[4] = {static_cast<int>(bgr.step[0]), 0, 0, 0};
        sws_scale(sws, in_data, in_linesize, 0, height, frame->data, frame->linesize);

        frame->pts = static_cast<int64_t>(frame_index);

        if (avcodec_send_frame(enc, frame) < 0)
            break;

        while (avcodec_receive_packet(enc, pkt) == 0)
        {
            bool is_key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;

            auto handle_packet = [&](AVPacket *p)
            {
                uint64_t pts90k = frame_index * 90000ull / static_cast<uint64_t>(fps);
                uint32_t ts32 = static_cast<uint32_t>(pts90k & 0xFFFFFFFFu);

                std::vector<uint8_t> ps = build_ps_pack_header();
                if (!sent_psm || is_key)
                {
                    std::vector<uint8_t> psm = build_psm();
                    ps.insert(ps.end(), psm.begin(), psm.end());
                    sent_psm = true;
                }

                std::vector<uint8_t> pes = build_pes_video(pts90k, p->data, p->size);
                ps.insert(ps.end(), pes.begin(), pes.end());

                rtp_send_ps(sockfd, dst, seq, ssrc_u32_, ts32, ps.data(), ps.size());
            };

            if (bsf_ctx)
            {
                if (av_bsf_send_packet(bsf_ctx, pkt) == 0)
                {
                    while (av_bsf_receive_packet(bsf_ctx, out_pkt) == 0)
                    {
                        handle_packet(out_pkt);
                        av_packet_unref(out_pkt);
                    }
                }
            }
            else
            {
                handle_packet(pkt);
            }

            av_packet_unref(pkt);
        }

        frame_index++;
        next_tick += std::chrono::microseconds(frame_interval_us);
        std::this_thread::sleep_until(next_tick);
    }

    avcodec_send_frame(enc, nullptr);

    av_packet_free(&pkt);
    av_packet_free(&out_pkt);
    av_frame_free(&frame);
    sws_freeContext(sws);
    if (bsf_ctx)
        av_bsf_free(&bsf_ctx);
    avcodec_free_context(&enc);
    close(sockfd);
}
