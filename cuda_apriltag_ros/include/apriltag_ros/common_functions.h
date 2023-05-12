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
 *
 ** common_functions.h *********************************************************
 *
 * Wrapper classes for AprilTag standalone and bundle detection. Main function
 * is TagDetector::detectTags which wraps the call to core AprilTag 2
 * algorithm, apriltag_detector_detect().
 *
 * $Revision: 1.0 $
 * $Date: 2017/12/17 13:23:14 $
 * $Author: dmalyuta $
 *
 * Originator:        Danylo Malyuta, JPL
 ******************************************************************************/

#ifndef CUDA_APRILTAG_ROS_COMMON_FUNCTIONS_H
#define CUDA_APRILTAG_ROS_COMMON_FUNCTIONS_H

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <thread>

#include <ros/ros.h>
#include <ros/console.h>
#include <XmlRpcException.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "nvapriltags/nvAprilTags.h"

#include "apriltag_ros/common_functions.h"
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h"

namespace cuda_apriltag_ros
{

static bool initialize_cuda(unsigned int width, unsigned int height)
{
  /* Check unified memory support. */
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  if (!devProp.managedMemory) {
    ROS_ERROR("CUDA device does not support managed memory.");
    return false;
  }

  /* Allocate output buffer. */
  size_t size = width * height * 4 * sizeof(char);
  cudaMallocManaged(&cuda_out_buffer, size, cudaMemAttachGlobal);
  cudaDeviceSynchronize();
  return true;
}

static bool convert_nv_to_cpu_tags(apriltag_detection_t* input, nvAprilTagsID_t* output) {
  
}

class TagDetector
{
 private:
  // Detections sorting
  static int idComparison(const void* first, const void* second);

  // Remove detections of tags with the same ID
  void removeDuplicates();

  // AprilTag code's attributes
  std::string family_;
  nvAprilTagsFamily nv_family_;
  int threads_;
  double decimate_;
  double blur_;
  int refine_edges_;
  int debug_;
  unsigned int width_, height_;
  int max_hamming_distance_ = 2;  // Tunable, but really, 2 is a good choice. Values of >=3
                                  // consume prohibitively large amounts of memory, and otherwise
                                  // you want the largest value possible.

  // Apriltag ROS CPU
  apriltag_ros::TagDetector cpu_detector_;

  // AprilTag objects
  zarray_t *detections_;
  nvAprilTagsCameraIntrinsics_t* cam_instrinsics_;
  
  // CUDA objects
  unsigned char* cuda_out_buffer_;

  // Other members
  std::map<int, StandaloneTagDescription> standalone_tag_descriptions_;
  std::vector<TagBundleDescription > tag_bundle_descriptions_;
  bool remove_duplicates_;
  bool run_quietly_;
  bool publish_tf_;
  tf::TransformBroadcaster tf_pub_;

 public:

  TagDetector(ros::NodeHandle pnh);
  ~TagDetector();

  bool findStandaloneTagDescription(
      int id, StandaloneTagDescription*& descriptionContainer,
      bool printWarning = true);

  // Detect tags in an image
  apriltag_ros::AprilTagDetectionArray detectTags(
      const cv_bridge::CvImagePtr& image,
      const sensor_msgs::CameraInfoConstPtr& camera_info);

  // Draw the detected tags' outlines and payload values on the image
  void drawDetections(cv_bridge::CvImagePtr image);

  bool get_publish_tf() const { return publish_tf_; }
};

} // namespace cuda_apriltag_ros

#endif // CUDA_APRILTAG_ROS_COMMON_FUNCTIONS_H
