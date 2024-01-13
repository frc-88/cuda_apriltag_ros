#!/usr/bin/env python3
import argparse
import glob
import os

import cv2
import numpy as np
import yaml


def write_yaml_parameters(path, parameters):
    print(f"Wrote to {path}")
    with open(path, "w") as file:
        image_height = parameters["image_height"]
        image_width = parameters["image_width"]
        distortion_model = parameters["distortion_model"]
        distortion_coefficients = parameters["distortion_coefficients"]
        camera_matrix = parameters["camera_matrix"]
        rectification_matrix = parameters["rectification_matrix"]
        projection_matrix = parameters["projection_matrix"]
        camera_name = parameters["camera_name"]
        contents = f"""
image_width: {image_width}
image_height: {image_height}
camera_name: {camera_name}
camera_matrix:
  rows: {camera_matrix["rows"]}
  cols: {camera_matrix["cols"]}
  data: {camera_matrix["data"]}
distortion_model: {distortion_model}
distortion_coefficients:
  rows: {distortion_coefficients["rows"]}
  cols: {distortion_coefficients["cols"]}
  data: {distortion_coefficients["data"]}
rectification_matrix:
  rows: {rectification_matrix["rows"]}
  cols: {rectification_matrix["cols"]}
  data: {rectification_matrix["data"]}
projection_matrix:
  rows: {projection_matrix["rows"]}
  cols: {projection_matrix["cols"]}
  data: {projection_matrix["data"]}
"""
        file.write(contents)


def get_matrix_dict(matrix):
    data = matrix.flatten().tolist()
    if len(matrix.shape) == 1:
        rows = 1
        cols = matrix.shape[0]
    else:
        rows = matrix.shape[0]
        cols = matrix.shape[1]
    return {
        "data": data,
        "rows": rows,
        "cols": cols,
    }


def main():
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser("calibrate")
    parser.add_argument(
        "-d", "--dir", type=str, default=os.path.join(script_dir, "../images")
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="config file",
        default=os.path.join(script_dir, "../config/calibration.yaml"),
    )
    parser.add_argument(
        "-n", "--name", type=str, help="Name of camera", default="camera"
    )
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    read_directory = args.dir
    write_path = (
        os.path.join(read_directory, "camera.yaml")
        if len(args.output) == 0
        else args.output
    )
    assert write_path.endswith(".yaml") or write_path.endswith(".yml")
    square_size = config["square_size"]

    # Define the dimensions of checkerboard
    checkerboard = (config["board_width"], config["board_height"])

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    obj_points = []

    # Creating vector to store vectors of 2D points for each checkerboard image
    img_points = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size

    gray = None
    shape = None
    images = glob.glob(f"./{read_directory}/*.jpg")
    for filename in images:
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if shape is None:
            shape = gray.shape
        else:
            assert shape == gray.shape, "Images are not all the same size!"
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        success, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if success:
            obj_points.append(objp)
            # refining pixel coordinates for given 2d points.
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            img_points.append(refined_corners)
        else:
            print(f"Failed to find checkerboard in {filename}")

    assert gray is not None
    assert shape is not None
    success, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    if success:
        distortion_coeffs = dist[0]
        camera_matrix = mtx
        rectification_matrix = np.eye(3)
        projection = np.zeros((3, 4))
        projection[0:3, 0:3] = mtx
        projection_matrix = projection

        parameters = {
            "image_height": shape[0],
            "image_width": shape[1],
            "distortion_model": "plumb_bob",
            "distortion_coefficients": get_matrix_dict(distortion_coeffs),
            "camera_matrix": get_matrix_dict(camera_matrix),
            "rectification_matrix": get_matrix_dict(rectification_matrix),
            "projection_matrix": get_matrix_dict(projection_matrix),
            "camera_name": "camera",
        }
        write_yaml_parameters(write_path, parameters)

    else:
        print("Failed to compute camera parameters!")


if __name__ == "__main__":
    main()
