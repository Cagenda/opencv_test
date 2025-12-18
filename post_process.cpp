#include "post_process.h"
// YOLOv5s 默认锚框尺寸 (对应 640x640 模型)这里的顺序必须和输出层一一对应
// stride 8 (P3) -> 80x80 特征图
// stride 16 (P4) -> 40x40 特征图
// stride 32 (P5) -> 20x20 特征图
const int anchors[3][6] = {
    {10, 13, 16, 30, 33, 23},     // 第一层 (Index 0)
    {30, 61, 62, 45, 59, 119},    // 第二层 (Index 1)
    {116, 90, 156, 198, 373, 326} // 第三层 (Index 2)
};

struct Probarry
{
    float conf;
    int index;
};
std::vector<Probarry> prob;

int readLines(const char *filepath, std::vector<std::string> &labels_vector, int maxlines)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        perror("file opne failed");
        return -1;
    }
    //==============成功打开了文件
    std::string line; // 定义一个临时接收变量
    while (getline(file, line))
    {
        labels_vector.emplace_back(line);
        if (labels_vector.size() >= static_cast<size_t>(maxlines))
        {
            std::cout << "Read labels to vector end\n " << std::endl;
            break;
        }
    }
    return labels_vector.size();
}
int loadlabelname(const char *filepath, std::vector<std::string> &labels_vector, int maxlines)
{
    int line_num = readLines(filepath, labels_vector, maxlines); // 调用了readLines函数，执行了函数里的逻辑
    return 0;
}

// =========================================快速排序===================================
static void sort_descending(std::vector<Probarry> &prob)
{
    std::sort(prob.begin(), prob.end(),
              [](const Probarry &a, const Probarry &b)
              {
                  return a.conf > b.conf;
              });
}

//==================================================反量化与sigmod===================
// ===========================反量化
static float deqnt_int8_to_f32(int8_t int_num, int32_t zp, float scale) // 为什么要加上static，在这里zp和scale是在yolo.cpp中输出信息打印出来了
{
    float float_num = (float)(int_num - zp) * scale;
    return float_num;
}

//============================激活函数=======================
static float sigmod(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

//==============从每一个head的buf中提取中每一个格子的（anchor）框的信息
static inline float get_val_formhead(int8_t *data, int c, int h, int w, int H, int W, float scale, int32_t zp)
{
    int idx = (c * H + h) * W + w; // NCHW 下标
    int8_t q = data[idx];
    return deqnt_int8_to_f32(q, zp, scale); // 用你写好的反量化函数
}

//==============================求IOU=======================
static float IoU(const Detection &a, const Detection &b)
{
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni = area_a + area_b - inter;

    if (uni <= 0.0f)
    {
        return 0.0f;
    }
    return inter / uni;
}

//===========================NMS函数=========================
static void nms_per_class(std::vector<Detection> &dets, float iou_thresh)
{
    std::vector<Detection> result; // 只放最后留下来的框

    if (dets.empty())
        return;
    // 1. 按 score 从大到小排序
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b)
              {
                  return a.score > b.score;
              });
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++)
    {
        if (removed[i])
            continue;
        result.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); ++j)
        {
            if (removed[j])
                continue;
            if (dets[i].class_id != dets[j].class_id)
                continue; // 只对同一类做 NMS

            float iou = IoU(dets[i], dets[j]);
            if (iou > iou_thresh)
            {
                removed[j] = true; // j 被 i 干掉
            }
        }
    }
    dets.swap(result);
}

// RGA 负责把摄像头拍的 1920x1080 照片 缩放 (Resize) 成 640x640。

// NPU 吃进 640x640 进行计算。

// Post Process 拿到结果后，再把坐标从 640x640 映射 (Map) 回 1920x1080，这样你才能在原图上画准框。

int post_process()
{
    static int init = -1;
    if (init == -1)
    {
        loadlabelname(LABEL_PATH, labels_vector, OBJ_CLASS_NUM);
        // 遍历vector容器
        for (std::string &s : labels_vector)
        {
            std::cout << "label name: " << s << std::endl;
        }
        init = 0;
        // deqnt_int8_to_f32();
    }

    return 0;
}

//=========================yOLOv5后处理入口=====================
std::vector<Detection> yolov5_post_process(
    const rknn_output *outputs,                     // rknn_outputs_get 得到的输出数组
    const std::vector<rknn_tensor_attr> &out_attrs, // ✅ 改成引用 vector 对应的 tensor 属性数组（包含 scale / zp / dims）
    int out_num,                                    // 输出个数=3
    int model_w,                                    // yolo模型输入640
    int model_h,                                    // 640
    int img_w,                                      // 原图宽
    int img_h,                                      // 原图高
    float conf_thres,                               // 例如 0.25
    float nms_thres                                 // 例如 0.45
)
{
    // 用来存所有 head 解码出来的候选框
    std::vector<Detection> dets;

    for (size_t i = 0; i < out_num; i++)
    {
        const rknn_tensor_attr &attr = out_attrs[i]; // 先获取一下，tesnor的属性
        const rknn_output &out = outputs[i];
        /*rknn_output 这个名字本身是一个“结构体类型”；
        你写 rknn_output outputs[3]; 时，才创建了一个“长度为 3 的 rknn_output 数组”。
        在函数参数里 const rknn_output* outputs 是“指向这个数组首元素的指针”，你在里面用 outputs[n] 来访问第 n 个 head 的输出。*/

        int8_t *data = static_cast<int8_t *>(out.buf); // 当前 head 的 int8 原始数据指针
        // out.buf 里装的是：这个 head 上 所有格子、所有通道 的数据。

        int C = attr.dims[1]; //  255
        int H = attr.dims[2]; // 经过NPU处理后，图像的纵轴有多少网格（格子数）
        int W = attr.dims[3]; // 经过NPU处理后，图像的横轴有多少网格
        float scale = attr.scale;
        int32_t zp = attr.zp;

        float stride_x = (float)model_w / W; // 一个网格的长度
        float stride_y = (float)model_h / H; // 一个网格宽度

        // ===== 遍历当前 head 的所有网格 (H, W) 和 3 个 anchor =====
        for (size_t h = 0; h < H; h++)
        {
            for (size_t w = 0; w < W; w++)
            {
                for (size_t a = 0; a < 3; a++) // 每个格子有 3 个 anchor：a = 0,1,2
                {

                    // 针对当前 (h,w,a) 的“一个框”去 data 里取 85 个通道的值
                    // ==== 1. 计算这个 anchor 在 C 维上的起始通道号 ====
                    // 每个 anchor 占多少个通道：4(bbox) + 1(obj) + num_classes
                    const int num_classes = 80;                // 你的模型类别数（COCO 就是 80）
                    const int anchor_stride = num_classes + 5; // 85

                    int c_base = static_cast<int>(a) * anchor_stride; // 当前这个 anchor 的起始通道号 = a × 每个 anchor 占的通道数

                    // ==== 2. 这个 anchor 的 tx/ty/tw/th/obj 对应的通道号 ====
                    int c_tx = c_base + 0;
                    int c_ty = c_base + 1;
                    int c_tw = c_base + 2;
                    int c_th = c_base + 3;
                    int c_obj = c_base + 4;

                    // ==== 3.从data中 (c, h, w) 取出这 5 个值，并反量化成 float（）,注意此时这里的tx，ty...并不是在图片中的真实坐标 ====
                    float tx_raw = get_val_formhead(data, c_tx, h, w, H, W, scale, zp);
                    float ty_raw = get_val_formhead(data, c_ty, h, w, H, W, scale, zp);
                    float tw_raw = get_val_formhead(data, c_tw, h, w, H, W, scale, zp);
                    float th_raw = get_val_formhead(data, c_th, h, w, H, W, scale, zp);
                    float obj_raw = get_val_formhead(data, c_obj, h, w, H, W, scale, zp);
                    // ==== 4. 对 obj 做一次 sigmoid，把它变成 [0,1] 概率 ====
                    float obj = sigmod(obj_raw); // 这里调用你之前写好的激活函数
                    // 如果 obj 非常小，后面大概率都被丢掉，可以预筛选（可选）
                    if (obj < 1e-3f)
                    {
                        continue; // 这一个 anchor 直接跳过，处理下一个 a
                    }

                    // 5. 遍历类别通道，找到“最可能的那个类别”对当前 (h,w,a) 这一“候选框”，在它的 80 个类别输出里，找出「哪个类别最有可能」以及「这个最有可能的类别的概率是多少」
                    float best_cls_prob = 0.0f; // 记录当前 anchor 上最大的类别概率
                    int best_cls_id = -1;       // 对应的类别 ID

                    for (int cls = 0; cls < num_classes; ++cls)
                    {
                        // 这个类别在 C 维上的通道号：
                        // 前面 5 个是 tx,ty,tw,th,obj，所以类别从 c_base + 5 开始往后排
                        int c_cls = c_base + 5 + cls; // c_cls是类别编号（每一个anchor中的后面80个类别）

                        // 取出这个 (c_cls, h, w) 位置的 raw 值（先反量化）
                        float cls_raw = get_val_formhead(data, c_cls, h, w, H, W, scale, zp);
                        // 对类别做一次 sigmoid，把它变成 [0,1] 概率
                        float cls_prob = sigmod(cls_raw); // 这里用你自己的激活函数

                        // 更新“最可能的那个类别”
                        if (cls_prob > best_cls_prob)
                        {
                            best_cls_prob = cls_prob;
                            best_cls_id = cls;
                        }
                    }
                    // 6. 计算最终的置信度：score = obj_conf * class_conf
                    float score = obj * best_cls_prob;
                    // 如果综合置信度太低，就没必要再解码坐标了，直接丢掉这个框
                    if (score < conf_thres)
                    {
                        continue; // 跳过这个 anchor，处理下一个
                    }
                    // 走到这里：
                    // - 这个框有较高的 obj（有目标的可能性）
                    // - 也有较高的 best_cls_prob（属于某一类的可能性）
                    // - score 也 >= conf_thres，可以认为是“值得保留”的候选框

                    //============7.解码：tx,ty,tw,th先得到它们在 模型输入尺寸（640×640）上的坐标。=============
                    // 当前 head 的 stride；你前面已经算过：
                    float stride_x = (float)model_w / W; // 比如 8、16、32
                    float stride_y = (float)model_h / H;

                    // 1. 解码中心点（在 640×640 上）
                    float x_center = (sigmod(tx_raw) * 2.0f - 0.5f + w) * stride_x;
                    float y_center = (sigmod(ty_raw) * 2.0f - 0.5f + h) * stride_y;

                    // 2. 解码宽高，需要用到这个 head 对应的 anchor 尺寸
                    // ✅ 建议改成：
                    float aw = static_cast<float>(anchors[i][2 * a + 0]);
                    float ah = static_cast<float>(anchors[i][2 * a + 1]);

                    float tw_act = sigmod(tw_raw);
                    float th_act = sigmod(th_raw);

                    // YOLOv5 里常用的宽高解码形式（结构理解即可）
                    float bw = (tw_act * 2.0f) * (tw_act * 2.0f) * aw; // 预测框宽度
                    float bh = (th_act * 2.0f) * (th_act * 2.0f) * ah; // 预测框高度

                    // ==================8.由 “中心点 + 宽高” 转成左上角/右下角坐标（仍然是 640×640 坐标系）===============
                    float x1 = x_center - bw / 2.0f;
                    float y1 = y_center - bh / 2.0f;
                    float x2 = x_center + bw / 2.0f;
                    float y2 = y_center + bh / 2.0f;

                    // 简单做一下边界裁剪，防止越界（可选，但很常见）
                    if (x1 < 0)
                        x1 = 0;
                    if (y1 < 0)
                        y1 = 0;
                    if (x2 > model_w)
                        x2 = (float)model_w;
                    if (y2 > model_h)
                        y2 = (float)model_h;

                    // ==================9.把 640×640 上的坐标，映射回原始图像大小===============
                    float r_x = (float)img_w / (float)model_w; // 宽度缩放比例
                    float r_y = (float)img_h / (float)model_h; // 高度缩放比例
                    Detection det;
                    det.class_id = best_cls_id;
                    det.score = score;

                    // 先把坐标从模型输入尺度映射回原图尺度
                    det.x1 = x1 * r_x;
                    det.y1 = y1 * r_y;
                    det.x2 = x2 * r_x;
                    det.y2 = y2 * r_y;

                    // （可选）再做一遍边界裁剪，保证不出界
                    if (det.x1 < 0)
                        det.x1 = 0;
                    if (det.y1 < 0)
                        det.y1 = 0;
                    if (det.x2 > img_w)
                        det.x2 = (float)img_w;
                    if (det.y2 > img_h)
                        det.y2 = (float)img_h;

                    // 10. 把这个候选框保存到 dets 中，后面统一做 NMS
                    dets.push_back(det);
                }

                /* code */
            }
        }
    }
    // 11. 所有 head 的候选框都收集完了，在 dets 里统一做一次 NMS
    nms_per_class(dets, nms_thres);

    // 12. 把最终的结果返回给调用者
    return dets;
}
