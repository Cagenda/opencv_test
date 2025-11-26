#include "yolov5s.h"

Yolov5s::Yolov5s(const char *model_path, int npu_index)
{
    int ret;//接收init函数的返回值
    this->model_data_size = 0;
    this->model_data = load_model(model_path,&this->model_data_size);
    ret = rknn_init(&this->ctx,model_data,model_data_size,0,NULL);
    if(ret<0)
    {
        printf("model init failde \n");
    }
    else
    {
        printf("model init success \n");
    }
}
Yolov5s::~Yolov5s()
{
    if(model_data)
    {
        free(model_data);
    }
}



unsigned char *Yolov5s::load_model(const char *file_name, int *model_size)
{
    FILE *fp;
    unsigned char *data;
    fp = fopen(file_name, "rb"); // rb代表读取图片/视频等二进制文件
    if(fp==NULL)
    {
        perror("File open failed: ");
    }
    fseek(fp, 0, SEEK_END);//将光标位置跳转到文件末尾
    long size = ftell(fp);//ftell(fp) 返回“当前文件指针的位置（偏移）”，单位是字节，当前光标位置 = 文件大小（字节）
    *model_size = size;
    //开辟内存空间，设置偏移量，将文件内容（fp指向的数据）放入开辟的空间中
    data = load_data(fp,0,size);
    return data;
}
unsigned char *Yolov5s::load_data(FILE *fp, size_t offset, size_t sz)
{
    unsigned char *data;//用来接收malloc返回的地址
    int ret;//用来接收函数的返回值，
    if(fp==NULL)
    {
        return NULL;
    }
    ret = fseek(fp, offset, SEEK_SET);//判断是否读取成功，返回值为0，则读取成功
    if(ret!=0)
    {
        perror("load data fseek error");
    }
    data = (unsigned char*)malloc(sz);//开辟内存空间
    if(data==NULL)
    {
        perror("load data  malloc error");
    }
    ret = fread(data, 1, sz, fp);//将读取的数据（fp中的数据）放入内存空间
    if(ret==0)
    {
        perror("load data fread error");
    }

    return data;
}
