# 指定使用的最低 CMake 版本，以确保支持所有需要的功能和特性。
cmake_minimum_required(VERSION 3.5)
# 定义项目名称为 cv。CMake 将使用这个名称来创建相关构建目录和变量。
project(app)
# 设置编译器标准为 C++11。这将确保代码在 C++11 标准下进行编译。
set(CMAKE_CXX_STANDARD 11)
# 输出系统的基本信息，包括系统名称和版本号，便于调试和确认构建环境。
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
# 查找并引入 OpenCV 库。如果找不到，将会给出错误提示。
find_package(OpenCV REQUIRED)
# 输出 OpenCV 库的信息，包括版本、库路径和头文件路径，帮助确认配置是否正确。
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")       # 输出 OpenCV 的版本
message(STATUS "    c'mac'macmalibraries: ${OpenCV_LIBS}")       # 输出 OpenCV 链接的库
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")  # 输出 OpenCV 包含文件的路径
message(" this is meihua!")

#设定RK3588的库
set(DEVICE_NAME "RK3588")
#设定第三方库的
set(LIB_ARCH "aarch64")
set(3DPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RKNN_API_PATH ${3DPARTY_PATH}/librknn_api)
set(RKNN_INCLUDE_DIRS ${RKNN_API_PATH}/include)
set(RKNN_LIBS ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)


#rga_lib
set(RGA_DIR ${3DPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_INCLUDE_DIRS ${RGA_DIR}/include)
set(RGA_LIBS ${RGA_DIR}/lib/Linux/aarch64/librga.so)
# 将 OpenCV 的头文件路径添加到编译器的搜索路径。
# 这使得源文件可以正确包含 OpenCV 的头文件。
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
)

# 定义一个名为 cv 的可执行目标，该目标将从 main.cpp 源文件生成。
add_executable(app test1.cpp thread_pool.cpp yolov5s.cpp)
# 将 OpenCV 的库与目标可执行文件 cv 链接，确保在程序运行时能够调用 OpenCV 函数。
target_link_libraries(app ${OpenCV_LIBS}  ${RKNN_LIBS}  ${RGA_LIBS})