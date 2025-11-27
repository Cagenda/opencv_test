#include "yolov5s.h"

Yolov5s::Yolov5s(const char *model_path, int npu_index)
{
    int ret; // 接收函数的返回值
    this->model_data_size = 0;
    this->model_data = load_model(model_path, &this->model_data_size);
    ret = rknn_init(&this->ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("model init failde \n");
    }
    else
    {
        printf("model init success \n");
    }
    // 调用NPU
    rknn_core_mask core_mask;
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

    // 1.查询SDK_Version
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &this->version, sizeof(this->version)); // 把SDK_Version信息存放在version中
    printf("sdk version :%s,drv version:%s\n", version.api_version, version.drv_version);
    // 2.查询tensor num
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &this->io_num, sizeof(this->io_num));
    printf("input num :%d\n output num :%d\n", this->io_num.n_input, this->io_num.n_output);
    // 3.
    input_attr.resize(io_num.n_input); // 让 vector(就是input_attr)里有n个元素。每个元素都是rknn_tensor_attr() 的默认值。根据模型实际的输入/输出数量，创建相应大小的属性数组，用来存放所有 tensor 的信息。
    output_attr.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attr[i].index = i;
        ret = rknn_query(ctx,
                         RKNN_QUERY_INPUT_ATTR,
                         &(input_attr[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            perror("Yolov5s get index input failed");
        }
    }

    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attr[i].index = i;
        ret = rknn_query(ctx,
                         RKNN_QUERY_OUTPUT_ATTR,
                         &(output_attr[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            perror("Yolov5s get index output failed");
        }
    }
}

Yolov5s::~Yolov5s()
{
    if (model_data)
    {
        free(model_data);
    }
}

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
    *model_size = size;
    // 开辟内存空间，设置偏移量，将文件内容（fp指向的数据）放入开辟的空间中
    data = load_data(fp, 0, size);
    return data;
}
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
    ret = fread(data, 1, sz, fp); // 将读取的数据（fp中的数据）放入内存空间
    if (ret == 0)
    {
        perror("load data fread error");
    }
    return data;
}
