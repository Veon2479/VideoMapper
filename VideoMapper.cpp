#include <iostream>
#include "VideoMapper.h"

// TODO: add input checks for integer and double setters

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedValue"

void VideoMapper::setTryCuda(bool arg) {
    try_cuda = arg;
}

void VideoMapper::setPreview(bool arg) {
    preview = arg;
}

void VideoMapper::setWorkMegapix(double arg) {
    work_megapix = arg;
}

void VideoMapper::setConfThresh(float arg) {
    conf_tresh = arg;
}

void VideoMapper::setMatchConf(double arg) {
    match_conf = arg;
}

void VideoMapper::setBaMask(int arg) {
    if ((arg & NO_FX) == 0) refine_mask(0,0) = 1;
    if ((arg & NO_SKEW) == 0) refine_mask(0,1) = 1;
    if ((arg & NO_PPX) == 0) refine_mask(0,2) = 1;
    if ((arg & NO_ASPECT) == 0) refine_mask(1,1) = 1;
    if ((arg & NO_PPY) == 0) refine_mask(1,2) = 1;
}

void VideoMapper::setDoWaveCorrect(bool arg) {
    do_wave_correct = arg;
}

void VideoMapper::setWaveCorrection(cv::detail::WaveCorrectKind arg) {
    wave_correct = arg;
}

void VideoMapper::setSeamMegapix(double arg) {
    seam_megapix = arg;
}

void VideoMapper::setSeam(Seam arg) {
    switch (arg) {
        case Seam::Gc_color:
            seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
            break;
        case Seam::Gc_colorgrad:
            seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
            break;
        case Seam::Voronoi:
            seam_finder = cv::makePtr<cv::detail::VoronoiSeamFinder>();
            break;
        case Seam::No:
            seam_finder = cv::makePtr<cv::detail::NoSeamFinder>();
            break;
        case Seam::Dp_color:
            seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR);
            break;
        case Seam::Dp_colorgrad:
            seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR_GRAD);
            break;
    }
}

void VideoMapper::setComposeMegapix(double arg) {
    compose_megapix = arg;
}

void VideoMapper::setExpComp(int arg) {
    expos_comp_type = arg;
}

void VideoMapper::setExpCompNrFeeds(int arg) {
    expos_comp_nr_feeds = arg;
}

void VideoMapper::setExpCompNrFiltering(int arg) {
    expos_comp_nr_filtering = arg;
}

void VideoMapper::setExpCompBlockSize(int arg) {
    expos_comp_block_size = arg;
}

void VideoMapper::setBlendType(int arg) {
    blend_type = arg;
}

void VideoMapper::setBlendStrength(int arg) {
    blend_strength = arg;
}

VideoMapper::~VideoMapper() = default;

VideoMapper::VideoMapper() {
    setTryCuda(false);
    setPreview(false);

    setWorkMegapix(0.6);
    setMatchConf(0.65);
    setConfThresh(1.0);
    setBaMask(0);
    setDoWaveCorrect(false);
    setWaveCorrection(cv::detail::WAVE_CORRECT_HORIZ);

    setSeam(Seam::Gc_color);
    setSeamMegapix(0.1);
    setComposeMegapix(-1);
    setExpComp(cv::detail::ExposureCompensator::GAIN_BLOCKS);
    setExpCompNrFeeds(1);
    setExpCompNrFiltering(2);
    setExpCompBlockSize(32);
    setBlendType(cv::detail::Blender::MULTI_BAND);
    setBlendStrength(5);

    finder = cv::xfeatures2d::SURF::create();
    matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
    adjuster = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();
    warper_creator = cv::makePtr<cv::AffineWarper>();
}

cv::Mat VideoMapper::makeMapFromVideo(cv::VideoCapture cap, int frameCount, int frameDelta, int startOffset) {
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    bool isSuccess = true;
    while (isSuccess && cap.isOpened() && startOffset > 0)
    {
        isSuccess = cap.read(frame);
    }
    if (!isSuccess || !cap.isOpened())
        throw std::invalid_argument("Error while reaching startOffset in the video");

    int i = 1;
    if (frameCount == -1)
        frameCount = std::numeric_limits<int>::max();;
    while (isSuccess && cap.isOpened() && frameCount > 0)
    {
        isSuccess = cap.read(frame);
        if (isSuccess && (i % frameDelta == 0))
        {
            frames.push_back(frame);
            frameCount--;
        }
        frame.release();
        i++;
    }
    return makeMapFromImages(frames);
}

cv::Mat VideoMapper::makeMapFromImages(const std::vector<cv::Mat>& Images) {
    if (Images.size() < 2)
        throw std::invalid_argument("Not enough images in the input vector");

    double work_scale = 1, compose_scale = 1;
    if (work_megapix >= 0)
    {
        work_scale = std::min(1.0, sqrt(work_megapix * 1e6 / Images[0].size().area()));
    }
    double seam_scale = std::min(1.0, sqrt(seam_megapix * 1e6 / Images[0].size().area()));
    double seam_work_aspect = seam_scale / work_scale;

    // Finding features
    cv::Mat img;
    std::vector<cv::detail::ImageFeatures> features(Images.size());
    std::vector<cv::Mat> images(Images.size());
    std::vector<cv::Size> full_img_sizes(Images.size());
    for (int i = 0; i < Images.size(); i++)
    {
        if (Images[i].empty())
            throw std::invalid_argument("Empty frame detected at position " + std::to_string(i));
        full_img_sizes[i] = Images[i].size();
        cv::resize(Images[i], img, cv::Size(), work_scale, work_scale, cv::INTER_LINEAR_EXACT);
        cv::detail::computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        cv::resize(Images[i], img, cv::Size(), seam_scale, seam_scale, cv::INTER_LINEAR_EXACT);
        images[i] = img.clone();
        img.release();
    }

    // Searching for matches
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

//    // Do some intrusion in matching results
//    const int count = static_cast<int>(pairwise_matches.size());
//    for (int i = 0; i < count; i++)
//    {
//        if (abs(pairwise_matches[i].src_img_idx - pairwise_matches[i].dst_img_idx) == 1)
//            pairwise_matches[i].confidence = conf_tresh - 0.1;
//        else
//            pairwise_matches[i].confidence /= 2;
//    }

    // TODO: add flag for not rejection of any frame
    // Reject some frames that has not enough matches with others
    std::vector<int> indices = cv::detail::leaveBiggestComponent(features, pairwise_matches, static_cast<float>(conf_tresh));
    std::vector<cv::Mat> img_subset;
    std::vector<cv::Size> full_img_size_subset;
    for (size_t i = 0; i < indices.size(); i++)
    {
        img_subset.push_back(images[indices[i]].clone());
        images[indices[i]].release();
        full_img_size_subset.push_back(full_img_sizes[indices[i]]);
    }
    for (auto item : images)
        if (!item.empty())
            item.release();
    images.clear();
    images = img_subset;
    full_img_sizes.clear();
    full_img_sizes = full_img_size_subset;
    if (images.size() < 2)
        throw std::invalid_argument("Not enough frames after rejection inappropriate ones");

    std::vector<cv::detail::CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
        throw std::invalid_argument("Homography estimation failed");

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cv::Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    // TODO: check if it is possible while using affine estimator and other affine things
    // Adjusting camera parameters
    adjuster->setConfThresh(conf_tresh);
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
        throw std::invalid_argument("Camera parameters adjusting failed");

    // Find median focal length
    std::vector<double> focals;
    for (auto & camera : cameras)
        focals.push_back(camera.focal);
    std::sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    // Do wave correction
    if (do_wave_correct)
    {
        std::vector<cv::Mat> rmats;
        for (auto & camera : cameras)
            rmats.push_back(camera.R.clone());
        cv::detail::waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    // Do auxiliary warping images
    std::vector<cv::Point> corners(images.size());
    std::vector<cv::UMat> masks_warped(images.size());
    std::vector<cv::UMat> images_warped(images.size());
    std::vector<cv::Size> sizes(images.size());
    std::vector<cv::UMat> masks(images.size());

    // Prepare images masks
    for (size_t i = 0; i < images.size(); ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(cv::Scalar::all(255));
    }

    // Warp images and their masks
    cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    auto swa = (float)seam_work_aspect;
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;
        corners[i] = warper->warp(images[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
    }

    std::vector<cv::UMat> images_warped_f(images.size());
    for (size_t i = 0; i < images.size(); ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    // Do compensate exposure
    cv::Ptr<cv::detail::ExposureCompensator> compensator = cv::detail::ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<cv::detail::GainCompensator*>(compensator.get()))
    {
        cv::detail::GainCompensator* gcompensator = dynamic_cast<cv::detail::GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get()))
    {
        cv::detail::ChannelsCompensator* ccompensator = dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get()))
    {
        cv::detail::BlocksCompensator* bcompensator = dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }
    compensator->feed(corners, images_warped, masks_warped);

    // Find seams
    seam_finder->find(images_warped_f, corners, masks_warped);

//     Release unused resources
    for (size_t i = 0; i < images.size(); ++i)
    {
        images[i].release();
        images_warped_f[i].release();
        images_warped[i].release();
        masks[i].release();
    }
    images.clear();
    images_warped_f.clear();
    images_warped.clear();
    masks.clear();

    // Do compositing
    cv::Mat img_warped, img_warped_s, full_img;
    full_img = Images[0].clone();
    full_img.release();
    cv::Mat dilated_mask, seam_mask, mask, mask_warped;
    double compose_work_aspect = 1;
    cv::Ptr<cv::detail::Blender> blender;
    bool is_compose_scale_set = false;

    for (int i = 0; i < indices.size(); ++i)
    {
        full_img = Images[indices[i]];
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = std::min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (size_t j = 0; j < indices.size(); ++j)
            {
                // Update intrinsics
                cameras[j].focal *= compose_work_aspect;
                cameras[j].ppx *= compose_work_aspect;
                cameras[j].ppy *= compose_work_aspect;

                // Update corner and size
                cv::Size sz = full_img_sizes[j];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[j].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[j].height * compose_scale);
                }

                cv::Mat K;
                cameras[j].K().convertTo(K, CV_32F);
                cv::Rect roi = warper->warpRoi(sz, K, cameras[j].R);
                corners[j] = roi.tl();
                sizes[j] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR_EXACT);
        else
            img = full_img.clone();
        full_img.release();
        cv::Size img_size = img.size();

        cv::Mat K;
        cameras[i].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(cv::Scalar::all(255));
        warper->warp(mask, K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(i, corners[i], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[i], dilated_mask, cv::Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, cv::INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender)
        {
            blender = cv::detail::Blender::createDefault(blend_type, try_cuda);
            cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
            float blend_width = std::sqrt(static_cast<float>(dst_sz.area())) * (float) blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, try_cuda);
            else if (blend_type == cv::detail::Blender::MULTI_BAND)
            {
                cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(std::log(blend_width)/std::log(2.)) - 1.));
            }
            else if (blend_type == cv::detail::Blender::FEATHER)
            {
                cv::detail::FeatherBlender* fb = dynamic_cast<cv::detail::FeatherBlender*>(blender.get());
                fb->setSharpness(1.f/blend_width);
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, corners[i]);

    }

    cv::Mat result, result_mask;
    blender->blend(result, result_mask);

    return result;
}



#pragma clang diagnostic pop