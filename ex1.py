import os
import shutil
import random

def pick_and_copy_images(src_folder, dest_folder, percentage=0.2):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all files in the source folder
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    # Shuffle the list of files
    random.shuffle(files)

    # Calculate the number of files to select
    num_files_to_select = int(len(files) * percentage)

    # Select the files
    selected_files = files[:num_files_to_select]

    # Copy the selected files to the destination folder
    for file in selected_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(dest_folder, file))

import cv2
import os
import numpy as np
from tqdm import tqdm

calib_board_img_path = '/home/hussain/Projects/Personal/camera_calibration/generate_charuco_boards/new_images/DICT_4X4_50_5_3_0.03_0.015_2000_img.png'

overly_thickness = 1

corner_colors_rgb = np.array([(255,0,0), (0,255,0), (0,0,255), (255,255,0)])[:, ::-1]
draw_colors_rgb = np.array([(230, 25, 75), (60, 180, 75), (255, 255, 25),(0, 130, 200),(245, 130, 48),
                            (145, 30, 180), (70, 240, 240), (240, 50, 230),(210, 245, 60),(250, 190, 190),
                            (0,128,128), (230, 190, 255), (170, 110,40), (250, 250, 200), (128, 0, 0),
                            (170, 255, 195), (128, 128, 0), (0, 0, 128), (128, 128, 128), (255, 255, 255),
                            (0, 0, 0)])[:, ::-1]
max_colors = len(draw_colors_rgb)

# ------------------------------
# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 3
SQUARE_LENGTH = 25/1000 #0.01
MARKER_LENGTH = 12/1000 #0.008
# ...


def viz_aruco_dets_poly_all(cam_img, marker_ids, marker_corners):
    sort_idxs = np.argsort(marker_ids[:, 0])
    sorted_mIDs = marker_ids[sort_idxs, 0]
    sorted_m_corners = [marker_corners[s_idx] for s_idx in sort_idxs]
    num_markers = len(sorted_mIDs)
    ovrly_img = cam_img.copy()

    for m_idx in tqdm(range(num_markers)):

        c_idx = m_idx % max_colors
        color = (int(draw_colors_rgb[c_idx, 0]), int(draw_colors_rgb[c_idx, 1]), int(draw_colors_rgb[c_idx, 2]))
        cv2.polylines(ovrly_img, sorted_m_corners[m_idx].astype(np.int32), True, color, overly_thickness, cv2.LINE_AA)
        center1 = np.mean(sorted_m_corners[m_idx][0, :, :], axis=0).astype('int')

        cv2.putText(ovrly_img, str(sorted_mIDs[m_idx]), center1, 6, 5, color, 4, cv2.LINE_AA)

        return ovrly_img

def viz_aruco_dets_corners_all(cam_img, marker_ids, marker_corners):

    sort_idxs = np.argsort(marker_ids[:, 0])
    sorted_mIDs = marker_ids[sort_idxs, 0]
    sorted_m_corners = [marker_corners[s_idx] for s_idx in sort_idxs]

    num_markers = len(sorted_mIDs)
    ovrly_img = cam_img.copy()

    for m_idx in tqdm(range(num_markers)):
        for c_idx in range(4):

            c_idx = m_idx % max_colors
            cv2.polylines(ovrly_img, sorted_m_corners[m_idx].astypenp.int32, True, color, overly_thickness, cv2.LINE_AA)

            center = sorted_m_corners[m_idx][0, c_idx, :].astype(int)
            color = (int(draw_colors_rgb[c_idx, 0]), int(draw_colors_rgb[c_idx, 1]), int(draw_colors_rgb[c_idx, 2]))

            cv2.circle(ovrly_img, center, 2, color, -1, cv2.LINE_AA)
        c_idx = m_idx % max_colors
        color = (int(draw_colors_rgb[c_idx, 0]), int(draw_colors_rgb[c_idx, 1]), int(draw_colors_rgb[c_idx, 2]))
        center = np.mean(sorted_m_corners[m_idx][0, :, :], axis=0).astype(int)
        cv2.putText(ovrly_img, str(sorted_mIDs[m_idx]), center, 2, 0.5, color, 1, cv2.LINE_AA)

        return ovrly_img

if __name__ == '__main__':

    board = cv2.aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)

    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerDistanceRate = 0.025
    parameters.adaptiveThreshWinSizeMin = 28
    parameters.adaptiveThreshWinSizeMax = 28
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.26

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, parameters)

    calib_img = cv2.imread(calib_board_img_path, cv2.IMREAD_GRAYSCALE)
    img_sz = calib_img.shape[:2]

    marker_corners_1, marker_ids_1, rej_cand_1 = detector.detectMarkers(calib_img)
    overly_img = viz_aruco_dets_poly_all(np.repeat(calib_img.reshape((img_sz[0], img_sz[1], 1)), 3, axis=2), marker_ids_1, marker_corners_1)

    out_filename = os.path.splitext(os.path.basename((calib_board_img_path)))[0]
    cv2.imwrite(f'{out_filename}_img.jpg', overly_img)


# Example usage
src_folder = 'path/to/source/folder'
dest_folder = 'path/to/destination/folder'
pick_and_copy_images(src_folder, dest_folder, percentage=0.2)
