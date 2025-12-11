#include "post_process.h"
struct Probarry
{
    float conf;
    int index;
};
std::vector<Probarry> prob;

std::vector<std::string> labels_vector; // 装labels的容器
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


//=================反量化与sigmod===================
//反量化
static float deqnt_int8_to_f32(int8_t int_num,int32_t zp,float scale)//为什么要加上fstatic，在这里zp和scale是在yolo.cpp中输出信息打印出来了
{
    float float_num = (float)(int_num-zp)*scale;
    return float_num;
}

//快速排序
static void sort_descending(std::vector<Probarry> &prob)
{
    std::sort(prob.begin(), prob.end(),
              [](const Probarry &a, const Probarry &b)
              {
                  return a.conf > b.conf;
              }
    );
}
//
int process()
{
    
}


//
int post_process()
{
    static int init = -1;
    if(init==-1)
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