#include "post_process.h"
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
int post_process()
{
    loadlabelname(LABEL_PATH, labels_vector, OBJ_CLASS_NUM);
    // 遍历vector容器
    for (std::string &s : labels_vector)
    {
        std::cout << "label name: " << s << std::endl;
    }
    return 0;
}