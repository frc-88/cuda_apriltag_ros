/**
 * Copyright (c) 2017, California Institute of Technology.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the California Institute of
 * Technology.
 */

#include "cuda_apriltag_ros/common_functions.h"
#include "image_geometry/pinhole_camera_model.h"

#include "common/homography.h"

#include "tag36h11.h"  // TODO: delete
#include "tag16h5.h"  // TODO: delete

namespace cuda_apriltag_ros
{

TagDetector::TagDetector(ros::NodeHandle pnh) :
    family_(apriltag_ros::getAprilTagOption<std::string>(pnh, "tag_family", "tag36h11")),
    decimate_(apriltag_ros::getAprilTagOption<double>(pnh, "tag_decimate", 1.0)),
    blur_(apriltag_ros::getAprilTagOption<double>(pnh, "tag_blur", 0.0)),
    refine_edges_(apriltag_ros::getAprilTagOption<int>(pnh, "tag_refine_edges", 1)),
    debug_(apriltag_ros::getAprilTagOption<int>(pnh, "tag_debug", 0)),
    max_hamming_distance_(apriltag_ros::getAprilTagOption<int>(pnh, "max_hamming_dist", 2)),
    publish_tf_(apriltag_ros::getAprilTagOption<bool>(pnh, "publish_tf", false)),
    max_tags_(apriltag_ros::getAprilTagOption<int>(pnh, "max_tags", 100))
{
  cpu_detector_ = std::shared_ptr<apriltag_ros::TagDetector>(new apriltag_ros::TagDetector(pnh));

  // Parse standalone tag descriptions specified by user (stored on ROS
  // parameter server)
  XmlRpc::XmlRpcValue standalone_tag_descriptions;
  if(!pnh.getParam("standalone_tags", standalone_tag_descriptions))
  {
    ROS_WARN("No april tags specified");
  }
  else
  {
    try
    {
      standalone_tag_descriptions_ =
          cpu_detector_->parseStandaloneTags(standalone_tag_descriptions);
    }
    catch(XmlRpc::XmlRpcException e)
    {
      // in case any of the asserts in parseStandaloneTags() fail
      ROS_ERROR_STREAM("Error loading standalone tag descriptions: " <<
                       e.getMessage().c_str());
    }
  }

  // parse tag bundle descriptions specified by user (stored on ROS parameter
  // server)
  XmlRpc::XmlRpcValue tag_bundle_descriptions;
  if(!pnh.getParam("tag_bundles", tag_bundle_descriptions))
  {
    ROS_WARN("No tag bundles specified");
  }
  else
  {
    try
    {
      tag_bundle_descriptions_ = cpu_detector_->parseTagBundles(tag_bundle_descriptions);
    }
    catch(XmlRpc::XmlRpcException e)
    {
      // In case any of the asserts in parseStandaloneTags() fail
      ROS_ERROR_STREAM("Error loading tag bundle descriptions: " <<
                       e.getMessage().c_str());
    }
  }

  // Optionally remove duplicate detections in scene. Defaults to removing
  if(!pnh.getParam("remove_duplicates", remove_duplicates_))
  {
    ROS_WARN("remove_duplicates parameter not provided. Defaulting to true");
    remove_duplicates_ = true;
  }

  // Define the tag family whose tags should be searched for in the camera
  // images
  nvAprilTagsFamily nv_family_;
  if (family_ == "tag36h11")
  {
    nv_family_ = NVAT_TAG36H11;
    tf_ = tag36h11_create();  // TODO: delete
  }
  else if (family_ == "tag16h5")
  {
    nv_family_ = NVAT_TAG16H5;
    tf_ = tag16h5_create();  // TODO: delete
  }
  else
  {
    throw std::runtime_error("Invalid tag family specified! Aborting");
  }

  cam_instrinsics_ = nullptr;
  cuda_out_buffer_ = NULL;
  detections_= NULL;
  main_stream_ = new cudaStream_t;
  is_initialized = false;

  // Create the AprilTag detector
  // TODO: delete
  td_ = apriltag_detector_create();
  apriltag_detector_add_family_bits(td_, tf_, max_hamming_distance_);
  td_->quad_decimate = (float)decimate_;
  td_->quad_sigma = (float)blur_;
  td_->nthreads = threads_;
  td_->debug = debug_;
  td_->refine_edges = refine_edges_;
}

// destructor
TagDetector::~TagDetector() {
  // Free memory associated with the array of tag detections
  if(detections_)
  {
    apriltag_detections_destroy(detections_);
  }

  // free memory associated with tag family
  if (family_ == "tag36h11")
  {
    // TODO
  }
  else if (family_ == "tag16h5")
  {
    // TODO
  }
}

apriltag_ros::AprilTagDetectionArray TagDetector::detectTags (
    const cv_bridge::CvImagePtr& image,
    const sensor_msgs::CameraInfoConstPtr& camera_info) {
  // Convert image to AprilTag code's format
  cv::Mat gray_image;
  if (image->image.channels() == 1)
  {
    gray_image = image->image;
  }
  else
  {
    cv::cvtColor(image->image, gray_image, CV_BGR2GRAY);
  }

  image_geometry::PinholeCameraModel camera_model;
  camera_model.fromCameraInfo(camera_info);

  // Get camera intrinsic properties for rectified image.
  double fx = camera_model.fx(); // focal length in camera x-direction [px]
  double fy = camera_model.fy(); // focal length in camera y-direction [px]
  double cx = camera_model.cx(); // optical center x-coordinate [px]
  double cy = camera_model.cy(); // optical center y-coordinate [px]

  ROS_INFO_STREAM_ONCE("Camera model: fx = " << fx << ", fy = " << fy << ", cx = " << cx << ", cy = " << cy);

  // Check if camera intrinsics are not available - if not the calculated
  // transforms are meaningless.
  if (fx == 0 && fy == 0) {
    ROS_WARN_STREAM_THROTTLE(5, "fx and fy are zero. Are the camera intrinsics set?");
    return apriltag_ros::AprilTagDetectionArray();
  }

  unsigned int width = gray_image.cols;
  unsigned int height = gray_image.rows;

  if (cam_instrinsics_ == nullptr) {
    cam_instrinsics_ = new nvAprilTagsCameraIntrinsics_t;
    cam_instrinsics_->fx = fx;
    cam_instrinsics_->fy = fy;
    cam_instrinsics_->cx = cx;
    cam_instrinsics_->cy = cy;
    width_ = width;
    height_ = height;
  }
  else {
    if (width != width_ || height != height_) {
      ROS_WARN_STREAM_THROTTLE(1, "Current image dimensions don't match the initial image. Not running apriltag detections.");
      return apriltag_ros::AprilTagDetectionArray();
    }
  }

  // If detectors are not initialized
  //  Get all tag sizes in standalones and bundles
  //  Initialize as many nv apriltag detectors as there are unique sizes (nvCreateAprilTagsDetector)
  //  Map tag ID to size
  // Else
  //  For each tag size
  //    run nvAprilTagsDetect
  //  For each detection
  //    Get expected size from ID to size map
  //    convert nvAprilTagsID_t to apriltag_detection_t. convert_nv_to_cpu_tags
  //    append detection to detections_

  if (is_initialized) {
    is_initialized = true;
    /* Check unified memory support. */
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    if (!devProp.managedMemory) {
      ROS_WARN("CUDA device does not support managed memory.");
    }

    /* Allocate output buffer. */
    size_t image_size = width_ * height_ * 4 * sizeof(char);
    ROS_INFO("Initializing CUDA with image size %dx%d", width_, height_);
    cudaMallocManaged(&cuda_out_buffer_, image_size, cudaMemAttachGlobal);
    cudaDeviceSynchronize();

    gpu_mat_ = cv::cuda::GpuMat(height_, width_, CV_8UC4, &cuda_out_buffer_);

    std::set<double> sizes;
    for (auto entry : standalone_tag_descriptions_) {
      apriltag_ros::StandaloneTagDescription description = entry.second;
      double size = description.size();
      sizes.insert(size);
      id_to_size_map_[description.id()] = size;
    }
    for (apriltag_ros::TagBundleDescription bundle : tag_bundle_descriptions_) {
      std::vector<int> ids = bundle.bundleIds();
      std::vector<double> bundle_sizes = bundle.bundleSizes();
      for (size_t index = 0; index < ids.size(); index++) {
        sizes.insert(bundle_sizes[index]);
        id_to_size_map_[ids[index]] = bundle_sizes[index];
      }
    }

    for (double size : sizes) {
      int error_code = nvCreateAprilTagsDetector(handles_[size], width_, height_, nv_family_, cam_instrinsics_, (float)size);
      if (error_code != 0) {
        throw std::runtime_error("Failed to initialize NV Apriltag detector with code: " + std::to_string(error_code));
      }
    }
    ROS_INFO("CUDA is initialized");
    cudaStreamCreate(main_stream_);
  }

  cuda_out_buffer_= gray_image.data;
  input_image_.dev_ptr = (uchar4*)cuda_out_buffer_;
  input_image_.pitch = gpu_mat_.step;
  std::vector<nvAprilTagsID_t> all_tags;
  ROS_INFO("mem async 1");
  cudaStreamAttachMemAsync(*main_stream_, input_image_.dev_ptr, 0, cudaMemAttachGlobal);
  for (auto entry : handles_) {
    double size = entry.first;
    nvAprilTagsHandle* april_tags_handle = handles_[size];

    std::vector<nvAprilTagsID_t> tags;
    uint32_t num_detections;
    const int error_code = nvAprilTagsDetect(
      *april_tags_handle, &input_image_, tags.data(),
      &num_detections, max_tags_, *main_stream_);
    if (error_code != 0) {
      throw std::runtime_error("Failed to run NV Apriltag detection with code: " + std::to_string(error_code));
    }

    for (nvAprilTagsID_t tag : tags) {
      if (id_to_size_map_[tag.id] == size) {
        all_tags.push_back(tag);
      }
    }
  }
  ROS_INFO("mem async 2");
  cudaStreamAttachMemAsync(*main_stream_, input_image_.dev_ptr, 0, cudaMemAttachHost);
  cudaStreamSynchronize(*main_stream_);


  // if (detections_)
  // {
  //   apriltag_detections_destroy(detections_);
  //   detections_ = zarray_create(all_tags.size());
  // }

  for (nvAprilTagsID_t tag : all_tags) {
    ROS_INFO("%d | hamming: %d",
      tag.id, tag.hamming_error
    );
    ROS_INFO("%d | corners:\n[[%f, %f], \n[%f, %f], \n[%f, %f], \n[%f, %f]]",
      tag.id,
      tag.corners[0].x,
      tag.corners[0].y,
      tag.corners[1].x,
      tag.corners[1].y,
      tag.corners[2].x,
      tag.corners[2].y,
      tag.corners[3].x,
      tag.corners[3].y
    );
    ROS_INFO("%d | rotate:\n[[%f, %f, %f], \n[%f, %f, %f], \n[%f, %f, %f]]",
      tag.id, 
      tag.orientation[0],
      tag.orientation[1],
      tag.orientation[2],
      tag.orientation[3],
      tag.orientation[4],
      tag.orientation[5],
      tag.orientation[6],
      tag.orientation[7],
      tag.orientation[8]
    );
    ROS_INFO("%d | translation:\n[%f, %f, %f]",
      tag.id, 
      tag.translation[0],
      tag.translation[1],
      tag.translation[2]
    );
  //   apriltag_detection_t detection = {};
  //   if (!convert_nv_to_apriltag_detection(&tag, &detection)) {
  //     ROS_WARN("Failed to convert tag");
  //     continue;
  //   }
  //   zarray_add(detections_, &detection);
  }

  // TODO: delete
  image_u8_t apriltag_image = { .width = gray_image.cols,
                                  .height = gray_image.rows,
                                  .stride = gray_image.cols,
                                  .buf = gray_image.data
  };
  if (detections_)
  {
    apriltag_detections_destroy(detections_);
    detections_ = NULL;
  }
  detections_ = apriltag_detector_detect(td_, &apriltag_image);
  for (int i=0; i < zarray_size(detections_); i++)
  {
    // Get the i-th detected tag
    apriltag_detection_t *detection;
    zarray_get(detections_, i, &detection);
    ROS_INFO("%d | center: [%f, %f] | hamming: %d",
      detection->id,
      detection->c[0],
      detection->c[1],
      detection->hamming
    );
    ROS_INFO("%d | corners:\n[[%f, %f], \n[%f, %f], \n[%f, %f], \n[%f, %f]]",
      detection->id,
      detection->p[0][0],
      detection->p[0][1],
      detection->p[1][0],
      detection->p[1][1],
      detection->p[2][0],
      detection->p[2][1],
      detection->p[3][0],
      detection->p[3][1]
    );
    ROS_INFO("%d | H:\n[[%f, %f, %f], \n[%f, %f, %f], \n[%f, %f, %f]]",
      detection->id,
      detection->H->data[0],
      detection->H->data[1],
      detection->H->data[2],
      detection->H->data[3],
      detection->H->data[4],
      detection->H->data[5],
      detection->H->data[6],
      detection->H->data[7],
      detection->H->data[8]
    );
  }

  // If remove_duplicates_ is set to true, then duplicate tags are not allowed.
  // Thus any duplicate tag IDs visible in the scene must include at least 1
  // erroneous detection. Remove any tags with duplicate IDs to ensure removal
  // of these erroneous detections
  if (remove_duplicates_)
  {
    removeDuplicates();
  }

  // Compute the estimated translation and rotation individually for each
  // detected tag
  apriltag_ros::AprilTagDetectionArray tag_detection_array;
  std::vector<std::string > detection_names;
  tag_detection_array.header = image->header;
  std::map<std::string, std::vector<cv::Point3d > > bundleObjectPoints;
  std::map<std::string, std::vector<cv::Point2d > > bundleImagePoints;
  for (int i=0; i < zarray_size(detections_); i++)
  {
    // Get the i-th detected tag
    apriltag_detection_t *detection;
    zarray_get(detections_, i, &detection);

    // Bootstrap this for loop to find this tag's description amongst
    // the tag bundles. If found, add its points to the bundle's set of
    // object-image corresponding points (tag corners) for cv::solvePnP.
    // Don't yet run cv::solvePnP on the bundles, though, since we're still in
    // the process of collecting all the object-image corresponding points
    int tagID = detection->id;
    bool is_part_of_bundle = false;
    for (unsigned int j=0; j<tag_bundle_descriptions_.size(); j++)
    {
      // Iterate over the registered bundles
      apriltag_ros::TagBundleDescription bundle = tag_bundle_descriptions_[j];

      if (bundle.id2idx_.find(tagID) != bundle.id2idx_.end())
      {
        // This detected tag belongs to the j-th tag bundle (its ID was found in
        // the bundle description)
        is_part_of_bundle = true;
        std::string bundleName = bundle.name();

        //===== Corner points in the world frame coordinates
        double s = bundle.memberSize(tagID)/2;
        cpu_detector_->addObjectPoints(s, bundle.memberT_oi(tagID),
                        bundleObjectPoints[bundleName]);

        //===== Corner points in the image frame coordinates
        cpu_detector_->addImagePoints(detection, bundleImagePoints[bundleName]);
      }
    }

    // Find this tag's description amongst the standalone tags
    // Print warning when a tag was found that is neither part of a
    // bundle nor standalone (thus it is a tag in the environment
    // which the user specified no description for, or Apriltags
    // misdetected a tag (bad ID or a false positive)).
    apriltag_ros::StandaloneTagDescription* standaloneDescription;
    if (!findStandaloneTagDescription(tagID, standaloneDescription,
                                      !is_part_of_bundle))
    {
      continue;
    }

    //=================================================================
    // The remainder of this for loop is concerned with standalone tag
    // poses!
    double tag_size = standaloneDescription->size();

    // Get estimated tag pose in the camera frame.
    //
    // Note on frames:
    // The raw AprilTag 2 uses the following frames:
    //   - camera frame: looking from behind the camera (like a
    //     photographer), x is right, y is up and z is towards you
    //     (i.e. the back of camera)
    //   - tag frame: looking straight at the tag (oriented correctly),
    //     x is right, y is down and z is away from you (into the tag).
    // But we want:
    //   - camera frame: looking from behind the camera (like a
    //     photographer), x is right, y is down and z is straight
    //     ahead
    //   - tag frame: looking straight at the tag (oriented correctly),
    //     x is right, y is up and z is towards you (out of the tag).
    // Using these frames together with cv::solvePnP directly avoids
    // AprilTag 2's frames altogether.
    // TODO solvePnP[Ransac] better?
    std::vector<cv::Point3d > standaloneTagObjectPoints;
    std::vector<cv::Point2d > standaloneTagImagePoints;
    cpu_detector_->addObjectPoints(tag_size/2, cv::Matx44d::eye(), standaloneTagObjectPoints);
    cpu_detector_->addImagePoints(detection, standaloneTagImagePoints);
    Eigen::Isometry3d transform = cpu_detector_->getRelativeTransform(standaloneTagObjectPoints,
                                                     standaloneTagImagePoints,
                                                     fx, fy, cx, cy);
    geometry_msgs::PoseWithCovarianceStamped tag_pose =
        cpu_detector_->makeTagPose(transform, image->header);

    // Add the detection to the back of the tag detection array
    apriltag_ros::AprilTagDetection tag_detection;
    tag_detection.pose = tag_pose;
    tag_detection.id.push_back(detection->id);
    tag_detection.size.push_back(tag_size);
    tag_detection_array.detections.push_back(tag_detection);
    detection_names.push_back(standaloneDescription->frame_name());
  }

  //=================================================================
  // Estimate bundle origin pose for each bundle in which at least one
  // member tag was detected

  for (unsigned int j=0; j<tag_bundle_descriptions_.size(); j++)
  {
    // Get bundle name
    std::string bundleName = tag_bundle_descriptions_[j].name();

    std::map<std::string,
             std::vector<cv::Point3d> >::iterator it =
        bundleObjectPoints.find(bundleName);
    if (it != bundleObjectPoints.end())
    {
      // Some member tags of this bundle were detected, get the bundle's
      // position!
      apriltag_ros::TagBundleDescription& bundle = tag_bundle_descriptions_[j];

      Eigen::Isometry3d transform =
          cpu_detector_->getRelativeTransform(bundleObjectPoints[bundleName],
                               bundleImagePoints[bundleName], fx, fy, cx, cy);
      geometry_msgs::PoseWithCovarianceStamped bundle_pose =
          cpu_detector_->makeTagPose(transform, image->header);

      // Add the detection to the back of the tag detection array
      apriltag_ros::AprilTagDetection tag_detection;
      tag_detection.pose = bundle_pose;
      tag_detection.id = bundle.bundleIds();
      tag_detection.size = bundle.bundleSizes();
      tag_detection_array.detections.push_back(tag_detection);
      detection_names.push_back(bundle.name());
    }
  }

  // If set, publish the transform /tf topic
  if (publish_tf_) {
    for (unsigned int i=0; i<tag_detection_array.detections.size(); i++) {
      geometry_msgs::PoseStamped pose;
      pose.pose = tag_detection_array.detections[i].pose.pose.pose;
      pose.header = tag_detection_array.detections[i].pose.header;
      tf::Stamped<tf::Transform> tag_transform;
      tf::poseStampedMsgToTF(pose, tag_transform);
      tf_pub_.sendTransform(tf::StampedTransform(tag_transform,
                                                 tag_transform.stamp_,
                                                 image->header.frame_id,
                                                 detection_names[i]));
    }
  }

  return tag_detection_array;
}

bool TagDetector::convert_nv_to_apriltag_detection(nvAprilTagsID_t* input, apriltag_detection_t* output)
{
  output->id = input->id;
  output->hamming = input->hamming_error;
  return true;
}

int TagDetector::idComparison (const void* first, const void* second)
{
  int id1 = (*(apriltag_detection_t**)first)->id;
  int id2 = (*(apriltag_detection_t**)second)->id;
  return (id1 < id2) ? -1 : ((id1 == id2) ? 0 : 1);
}

void TagDetector::removeDuplicates ()
{
  zarray_sort(detections_, &idComparison);
  int count = 0;
  bool duplicate_detected = false;
  while (true)
  {
    if (count > zarray_size(detections_)-1)
    {
      // The entire detection set was parsed
      return;
    }
    apriltag_detection_t *next_detection, *current_detection;
    zarray_get(detections_, count, &current_detection);
    int id_current = current_detection->id;
    // Default id_next value of -1 ensures that if the last detection
    // is a duplicated tag ID, it will get removed
    int id_next = -1;
    if (count < zarray_size(detections_)-1)
    {
      zarray_get(detections_, count+1, &next_detection);
      id_next = next_detection->id;
    }
    if (id_current == id_next || (id_current != id_next && duplicate_detected))
    {
      duplicate_detected = true;
      // Remove the current tag detection from detections array
      int shuffle = 0;
      apriltag_detection_destroy(current_detection);
      zarray_remove_index(detections_, count, shuffle);
      if (id_current != id_next)
      {
        ROS_WARN_STREAM("Pruning tag ID " << id_current << " because it "
                        "appears more than once in the image.");
        duplicate_detected = false; // Reset
      }
      continue;
    }
    else
    {
      count++;
    }
  }
}

void TagDetector::drawDetections (cv_bridge::CvImagePtr image)
{
  for (int i = 0; i < zarray_size(detections_); i++)
  {
    apriltag_detection_t *det;
    zarray_get(detections_, i, &det);

    // Check if this ID is present in config/tags.yaml
    // Check if is part of a tag bundle
    int tagID = det->id;
    bool is_part_of_bundle = false;
    for (unsigned int j=0; j<tag_bundle_descriptions_.size(); j++)
    {
      apriltag_ros::TagBundleDescription bundle = tag_bundle_descriptions_[j];
      if (bundle.id2idx_.find(tagID) != bundle.id2idx_.end())
      {
        is_part_of_bundle = true;
        break;
      }
    }
    // If not part of a bundle, check if defined as a standalone tag
    apriltag_ros::StandaloneTagDescription* standaloneDescription;
    if (!is_part_of_bundle &&
        !findStandaloneTagDescription(tagID, standaloneDescription, false))
    {
      // Neither a standalone tag nor part of a bundle, so this is a "rogue"
      // tag, skip it.
      continue;
    }

    // Draw tag outline with edge colors green, blue, blue, red
    // (going counter-clockwise, starting from lower-left corner in
    // tag coords). cv::Scalar(Blue, Green, Red) format for the edge
    // colors!
    line(image->image, cv::Point((int)det->p[0][0], (int)det->p[0][1]),
         cv::Point((int)det->p[1][0], (int)det->p[1][1]),
         cv::Scalar(0, 0xff, 0)); // green
    line(image->image, cv::Point((int)det->p[0][0], (int)det->p[0][1]),
         cv::Point((int)det->p[3][0], (int)det->p[3][1]),
         cv::Scalar(0, 0, 0xff)); // red
    line(image->image, cv::Point((int)det->p[1][0], (int)det->p[1][1]),
         cv::Point((int)det->p[2][0], (int)det->p[2][1]),
         cv::Scalar(0xff, 0, 0)); // blue
    line(image->image, cv::Point((int)det->p[2][0], (int)det->p[2][1]),
         cv::Point((int)det->p[3][0], (int)det->p[3][1]),
         cv::Scalar(0xff, 0, 0)); // blue

    // Print tag ID in the middle of the tag
    std::stringstream ss;
    ss << det->id;
    cv::String text = ss.str();
    int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 0.5;
    int baseline;
    cv::Size textsize = cv::getTextSize(text, fontface,
                                        fontscale, 2, &baseline);
    cv::putText(image->image, text,
                cv::Point((int)(det->c[0]-textsize.width/2),
                          (int)(det->c[1]+textsize.height/2)),
                fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
  }
}

bool TagDetector::findStandaloneTagDescription (
    int id, apriltag_ros::StandaloneTagDescription*& descriptionContainer, bool printWarning)
{
  std::map<int, apriltag_ros::StandaloneTagDescription>::iterator description_itr =
      standalone_tag_descriptions_.find(id);
  if (description_itr == standalone_tag_descriptions_.end())
  {
    if (printWarning)
    {
      ROS_WARN_THROTTLE(10.0, "Requested description of standalone tag ID [%d],"
                        " but no description was found...",id);
    }
    return false;
  }
  descriptionContainer = &(description_itr->second);
  return true;
}

} // namespace cuda_apriltag_ros
