#ifndef IMAGEMAPPERCPP_IMAGEMAPPER_H
#define IMAGEMAPPERCPP_IMAGEMAPPER_H

#include <opencv2/videoio.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/stitching/detail/blenders.hpp>
#include <stdexcept>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <cmath>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>


// <fx><skew><ppx><aspect><ppy>
enum BundleAdjustmentFlags : int {
    NO_PPY = 1,
    NO_ASPECT = 2,
    NO_PPX = 4,
    NO_SKEW = 8,
    NO_FX = 16
};

enum class Seam {
    No, Voronoi, Gc_color, Gc_colorgrad, Dp_color, Dp_colorgrad
};

class VideoMapper {
public:
    void setTryCuda(bool arg);
    void setPreview(bool arg);

    void setWorkMegapix(double arg);
    void setMatchConf(double arg);
    void setConfThresh(float arg);
    void setBaMask(int arg);
    void setDoWaveCorrect(bool arg);
    void setWaveCorrection(cv::detail::WaveCorrectKind arg);

    void setSeamMegapix(double arg);
    void setSeam(Seam arg);
    void setComposeMegapix(double arg);
    void setExpComp(int arg);
    void setExpCompNrFeeds(int arg);
    void setExpCompNrFiltering(int arg);
    void setExpCompBlockSize(int arg);
    void setBlendType(int arg);
    void setBlendStrength(int arg);

    cv::Mat makeMapFromImages(const std::vector<cv::Mat>& Images);
    cv::Mat makeMapFromVideo(cv::VideoCapture cap, int frameCount, int frameDelta = 1, int startOffset = 0);

    VideoMapper();
    ~VideoMapper();

private:
    // Flags
    bool try_cuda{};
    bool preview{};

    // Motion Estimation Flags
    double work_megapix{};
    double match_conf{};
    float conf_tresh{};
    bool do_wave_correct{};
    cv::detail::WaveCorrectKind wave_correct{};

    // Compositing flags
    double seam_megapix{};
    double compose_megapix{};
    int expos_comp_type{};
    int expos_comp_nr_feeds{};
    int expos_comp_nr_filtering{};
    int expos_comp_block_size{};
    int blend_type{};
    int blend_strength{};

    //runtime variables
    cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
    cv::Ptr<cv::Feature2D> finder = nullptr;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher = nullptr;
    cv::Ptr<cv::detail::Estimator> estimator = nullptr;
    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = nullptr;
    cv::Ptr<cv::WarperCreator> warper_creator = nullptr;
    cv::Ptr<cv::detail::SeamFinder> seam_finder = nullptr;
};


#endif //IMAGEMAPPERCPP_IMAGEMAPPER_H
