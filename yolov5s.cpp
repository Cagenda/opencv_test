#include "yolov5s.h"
static void print_tensor_attr(rknn_tensor_attr *attr)
{
    // dims[] 是数组，要打印得先变成一个字符串
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; i++)
    {
        std::string current_str = std::to_string(attr->dims[i]);
        shape_str = shape_str + " " + current_str;
    }
    // 打印tensor属性
    printf("index : %d, name : %s, n_dims = %d, dims = %s, size = %d, fmt = %d\n,scale=%f\n,zp=%d\n",
           attr->index,
           attr->name,
           attr->n_dims,
           shape_str.c_str(),
           attr->size,
           attr->fmt,
           attr->scale,
           attr->zp); // size：准备这么大的内存来装这个tensor,scale是缩放因子，zp 是零点 (Zero Point)。这是反量化公式中的 zp
}
int flag = 0;
//==========================================初始化函数
Yolov5s::Yolov5s(const char *model_path, int npu_index)
{ // 在这里需要先将模型放在RAM中，再从RAM中导入到NPU中

    //=========以下代码是将模型加载到RAM中,并且将模型加载到NPU，重点看load_model和load_data==============
    int ret; // 接收函数的返回值
    this->model_data_size = 0;
    this->model_data = load_model(model_path, &this->model_data_size); // 1.记录模型的数据和模型数据大小
    ret = rknn_init(&this->ctx, model_data, model_data_size, 0, NULL); // 2.把yolo模型加载到 NPU(此时还没有跑模型)
    if (ret < 0)
    {
        printf("model init failde \n");
    }
    else
    {
        printf("model init success \n");
    }
    //=============================================================================================
    // 调用NPU   一个 Yolov5s 实例 = 一个 rknn_context = 绑定到一个固定的 NPU 核。

    rknn_core_mask core_mask; // 设置 NPU 核（core_mask）
    if (npu_index == 0)
    {
        core_mask = RKNN_NPU_CORE_0;
    }
    else if (npu_index == 1)
    {
        core_mask = RKNN_NPU_CORE_1;
    }
    else
    {
        core_mask = RKNN_NPU_CORE_2;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        perror("rknn_init npu init failed");
        /* code */
    }

    //========以下内容是将查询模型的输入输出属性========

    // 1.查询SDK_Versisn（不知道用来做什么）把SDK_Version信息存放在version中
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &this->version, sizeof(this->version));
    // printf("sdk version :%s,drv version:%s\n", version.api_version, version.drv_version);

    // 2.查询tensor io_num,注意io_num是一个结构体
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &this->io_num, sizeof(this->io_num));
    // printf("input num :%d\n output num :%d\n", this->io_num.n_input, this->io_num.n_output);

    // 3.（获取 input_attr 和 output_attr)
    // io_num和其关系：io_num 告诉你“有多少个”，而 input_attr（或 inputs 数组）是用来“装这这么多东西”的容器。
    // 如果 io_num.n_input 的值为 2，那么 input_attr 这个容器（在你的代码中是一个 std::vector）里一定会有两个对应的元素。
    input_attr.resize(io_num.n_input);
    output_attr.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attr[i].index = i;
        ret = rknn_query(ctx,
                         RKNN_QUERY_INPUT_ATTR,
                         &(input_attr[i]),
                         sizeof(rknn_tensor_attr)); // 查询输入 tensor 属性。并且放入input_attr[i]中
        if (ret < 0)
        {
            perror("Yolov5s get index input failed");
        }
        // print_tensor_attr(&(input_attr[i]));
    }

    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attr[i].index = i;
        ret = rknn_query(ctx,
                         RKNN_QUERY_OUTPUT_ATTR,
                         &(output_attr[i]),
                         sizeof(rknn_tensor_attr)); // 查询输入 tensor 属性。并且放入output_attr[i]中
        if (ret < 0)
        {
            perror("Yolov5s get index output failed");
        }
        // print_tensor_attr(&(output_attr[i]));
    }
    // 获取输入format，对于NCHW和NHWC两种不一样的format分别讨论，信息写入变量中
    if (input_attr[0].fmt == RKNN_TENSOR_NCHW)
    {
        model_channel = input_attr[0].dims[1];
        model_height = input_attr[0].dims[2];
        model_weidth = input_attr[0].dims[3];
    }
    else if (input_attr[0].fmt == RKNN_TENSOR_NHWC)
    {
        model_height = input_attr[0].dims[1];
        model_weidth = input_attr[0].dims[2];
        model_channel = input_attr[0].dims[3];
    }
}
// =========================================析构函数
Yolov5s::~Yolov5s()
{
    if (model_data)
    {
        free(model_data);
    }
}
//=======================================推理函数部分
int Yolov5s::inference_image(const cv::Mat &orign_img, std::vector<Detection> &dets)
{
    //======================================图像根据模型输入输出信息进行预处理===================
    int ret = 0;
    img_weidth = orign_img.cols;
    img_height = orign_img.rows;
    img_channel = orign_img.channels();

    int resize_height = this->model_height;
    int resize_width = this->model_weidth;
    int resize_channel = this->model_channel;

    // printf("Image Width     :%d\n", img_weidth);
    // printf("Image Height    :%d\n", img_height);
    // printf("Image Channel   :%d\n", img_channel);

    // printf("Resize Width     :%d\n", resize_width);
    // printf("Resize Height    :%d\n", resize_height);
    // printf("Resize Channel   :%d\n", resize_channel);

    auto start = std::chrono::high_resolution_clock::now();
    // // opencv处理图像
    // cv::Mat img_cvt;    // 存放颜色空间转换后的图像（RGB）
    // cv::Mat img_resize; // 存放 resize 后的图像（模型输入大小）
    // cv::cvtColor(orign_img, img_cvt, cv::COLOR_BGR2RGB);
    // cv::resize(img_cvt, img_resize, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_AREA);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // printf("Opencv Process time:%ld   ms \n", duration.count());
    // cv::imwrite("img_cv_intera.jpg", img_resize);

    // -=========RGA进行图像处理=======================
    // 1.开辟内存空间
    char *src_buf, *src_cvt_buf, *dst_buf;
    src_buf = (char *)malloc(img_weidth * img_height * img_channel);         // 存放原始图数据
    src_cvt_buf = (char *)malloc(img_weidth * img_height * img_channel);     // 存放resize
    dst_buf = (char *)malloc(resize_width * resize_height * resize_channel); // 预处理后的像素数据缓冲区：一个连续的内存块，可以叫它 dst_buf

    // 2.地址初始化
    memcpy(src_buf, orign_img.data, img_weidth * img_height * img_channel); // 将原始图片数据copy进src_buf
    memset(src_cvt_buf, 0x00, img_weidth * img_height * img_channel);
    memset(dst_buf, 0x00, resize_width * resize_height * resize_channel);

    // 变为虚拟地址,把你的虚拟地址转换成 RGA 可用的地址
    rga_buffer_handle_t src_handle, src_cvt_handle, dst_handle;
    src_handle = importbuffer_virtualaddr(src_buf, img_weidth * img_height * img_channel);
    src_cvt_handle = importbuffer_virtualaddr(src_cvt_buf, img_weidth * img_height * img_channel);
    dst_handle = importbuffer_virtualaddr(dst_buf, resize_width * resize_height * resize_channel);
    // rga_buffer_handle_t src_handle = importbuffer_virtualaddr((void*)orign_img.data,  img_weidth * img_height * img_channel);
    if (src_handle == 0 || src_cvt_handle == 0 || dst_handle == 0)
    {
        printf("imppt va failed\n");
        return 1;
    }
    //
    rga_buffer_t src = wrapbuffer_handle(src_handle, img_weidth, img_height, RK_FORMAT_BGR_888);
    rga_buffer_t src_cvt = wrapbuffer_handle(src_cvt_handle, img_weidth, img_height, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_handle(dst_handle, resize_width, resize_height, RK_FORMAT_RGB_888);
    // 暂时不用检查 ret = imcheck(src, dst, {}, {});

    //  4.执行 (Resize + BGR转RGB)
    ret = imresize(src, dst);

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // printf("RGA Process time:%ld   ms \n", duration.count());

    // cv::Mat img_rga(resize_height, resize_width, CV_8UC3, dst_buf);
    // cv::imwrite("img_rga.jpg", img_rga);
    //======================================推理=======================================
    auto start2 = std::chrono::high_resolution_clock::now();
    int inputs_num = io_num.n_input;   // 获取rknn需要的输入节点数量。
    int outputs_num = io_num.n_output; // 获取rknn需要的输出节点数量。
    rknn_input inputs[inputs_num];     // 元素类型rknn_input。这是一个结构体,用来描述每一个输入张量的详细信息
    memset(inputs, 0, sizeof(inputs)); // 清空数组

    // =================填充第0个输入的信息=======
    inputs[0].index = 0;                                            //  1. 设置索引
    inputs[0].type = RKNN_TENSOR_UINT8;                             // 2. 设置数据类型
    inputs[0].size = resize_width * resize_height * resize_channel; // 3. 设置数据大小
    inputs[0].fmt = RKNN_TENSOR_NHWC;                               // 4. 设置数据排列格式（核心难点）RGA处理后的图像为NHWC，但是yolo需要NCHW
    inputs[0].pass_through = 0;                                     // 5. 自动转换从NHWC--->NHCW
    inputs[0].buf = dst_buf;                                        // 6. 挂载数据指针
    rknn_inputs_set(ctx, inputs_num, inputs);                       // 调用 rknn_inputs_set将数据拷贝到 NPU 输入内存
    //=================填充输出的信息==============
    rknn_output outputs[outputs_num];
    memset(outputs, 0, sizeof(outputs));
    // 循环填入输出信息
    for (int i = 0; i < outputs_num; i++)
    {
        outputs[i].want_float = 0; // 【关键】设为 0，不让驱动帮你把结果转成 float。原因：减少数据传输量（核心原因）int8 数据占用 1 个字节，而 float32 占用 4 个字节
    }
    //=================rknn_run===============================
    // printf("模型开始推理\n");
    ret = rknn_run(ctx, NULL); // 推理核心函数
    // if (ret == 0)
    // {
    //     printf("模型推理成功!\n");
    // }
    rknn_outputs_get(ctx, outputs_num, outputs, NULL);
    // auto end2 = std::chrono::high_resolution_clock::now();
    // auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    // printf("模型推理用时:%ld   ms \n", duration2.count());

    //========================================推理之后进行后处理=======================================
    dets = yolov5_post_process(
        outputs,
        output_attr, // &output_attr[0]
        outputs_num,
        640, 640,                       // model_w, model_h
        orign_img.cols, orign_img.rows, // img_w, img_h
        0.25f, 0.45f                    // conf阈值 / NMS阈值，你可以先用这对
    );

    //============================================释放空间==================================================================
    if (src_handle)
    {
        releasebuffer_handle(src_handle);
    }
    if (src_cvt_handle)
    {
        releasebuffer_handle(src_cvt_handle);
    }
    if (dst_handle)
    {
        releasebuffer_handle(dst_handle);
    }
    free(src_buf);
    free(dst_buf);
    free(src_cvt_buf);

    return ret;
}

// 把模型文件（.rknn）从文件系统读到内存，并返回这块内存的指针。
unsigned char *Yolov5s::load_model(const char *file_name, int *model_size)
{
    FILE *fp;
    unsigned char *data;
    fp = fopen(file_name, "rb"); // rb代表读取图片/视频等二进制文件
    if (fp == NULL)
    {
        perror("File open failed: ");
    }
    fseek(fp, 0, SEEK_END); // 将光标位置跳转到文件末尾
    long size = ftell(fp);  // ftell(fp) 返回“当前文件指针的位置（偏移）”，单位是字节，当前光标位置 = 文件大小（字节）
    *model_size = size;     // 将文件大小通过指针传出去

    // 开辟内存空间，设置偏移量，将文件内容（fp指向的数据）放入开辟的空间中
    data = load_data(fp, 0, size);
    return data;
}

// ✅把文件某一段内容按指定大小读到一块新分配的内存中（malloc），并返回这块内存的指针。
unsigned char *Yolov5s::load_data(FILE *fp, size_t offset, size_t sz)
{
    unsigned char *data; // 用来接收malloc返回的地址
    int ret;             // 用来接收函数的返回值，
    if (fp == NULL)
    {
        return NULL;
    }
    ret = fseek(fp, offset, SEEK_SET); // 判断是否读取成功，返回值为0，则读取成功
    if (ret != 0)
    {
        perror("load data fseek error");
    }
    data = (unsigned char *)malloc(sz); // 开辟内存空间
    if (data == NULL)
    {
        perror("load data  malloc error");
    }
    ret = fread(data, 1, sz, fp); // 将读取的数据（fp中的数据）放入内存（data）空间
    if (ret == 0)
    {
        perror("load data fread error");
    }
    return data;
}
