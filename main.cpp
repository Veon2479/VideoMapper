#include "VideoMapper.h"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

int main()
{
    auto mapper = VideoMapper();
    mapper.setMatchConf(0.3);
    mapper.setConfThresh(0.8);
    mapper.setWorkMegapix(0.1);
    mapper.setTryCuda(true);
    std::string name = "./1.mp4";
    cv::VideoCapture cap(name);
    auto res = mapper.makeMapFromVideo(cap, -1);
    cv::imwrite(name + ".png", res);
    return 0;
}