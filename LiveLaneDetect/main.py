# THIS IS ANTHONY FURCAL'S CODE FOR THE LANE LINE DETECTION ASSIGNMENT

import cv2
import numpy as np


# Getting the Camera
cap = cv2.VideoCapture('VideoFootage/car.mov')

def warp_frame(image, reverse):
   roi_points = np.array([
       (322, 174),  # Top-left corner
       (202, 237),  # Bottom-left corner
       (500, 237),  # Bottom-right corner
       (400, 174)  # Top-right corner
   ])


   dest_points = np.array([
       (0, 0),
       (0, 360),
       (640, 360),
       (640, 0)
   ])


   pts = np.float32(roi_points)
   pts2 = np.float32(dest_points)

   if not reverse:

        perspectiveMatrix = cv2.getPerspectiveTransform(pts, pts2)

   else:

       perspectiveMatrix = cv2.getPerspectiveTransform(pts2, pts)

   result = cv2.warpPerspective(image, perspectiveMatrix, (640, 360))


   return result



def hist_detection(binary_image, image):
    histogram = np.sum(binary_image[binary_image.shape[0] // 2:, :], axis=0)

    midpoint = np.int32(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9

    margin = 100

    minpix = 50

    window_height = np.int32(binary_image.shape[0] // nwindows)
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_image.shape[0] - (window + 1) * window_height
        win_y_high = binary_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        window_img = image
        margin = 100
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))
        result = window_img

        return result

    except TypeError:
        print("No line found")
        return image


def compass_overlay(image):
   overlay = cv2.imread('ExtraResources/CompasArrow.png', cv2.IMREAD_UNCHANGED)
   background = image


   if overlay.shape[2] == 4:
       b, g, r, alpha = cv2.split(overlay)
       alpha = alpha / 255.0
       overlay_rgb = cv2.merge([b, g, r])
   elif len(overlay.shape) == 2:
       alpha = overlay / 255.0
       overlay_rgb = overlay


   new_width = 100
   new_height = 100
   resized_overlay_rgb = cv2.resize(overlay_rgb, (new_width, new_height))
   resized_alpha = cv2.resize(alpha, (new_width, new_height))


   x_offset = 50
   y_offset = 50
   roi = background[y_offset:y_offset + new_height, x_offset:x_offset + new_width]


   roi_rgb = roi.astype(np.float32)
   overlay_rgb = resized_overlay_rgb.astype(np.float32)


   blended_roi = (
               roi_rgb * (1 - resized_alpha[:, :, np.newaxis]) + overlay_rgb * resized_alpha[:, :, np.newaxis]).astype(
       np.uint8)
   background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = blended_roi


   return background



def stream_processing():
  while cap.isOpened():
      ret, frame = cap.read()
      resized_frame = cv2.resize(frame, (640, 360))
      warped = warp_frame(resized_frame, False)
      overlay = compass_overlay(resized_frame)
      gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
      sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
      abs_sobelx = np.absolute(sobelx)
      scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
      sx_binary = np.zeros_like(scaled_sobel)
      sx_binary[(scaled_sobel >= 90) & (scaled_sobel <= 255)] = 1

      white_binary = np.zeros_like(gray)
      white_binary[(gray > 230) & (gray <= 255)] = 1

      hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
      H = hls[:, :, 0]
      S = hls[:, :, 2]

      sat_binary = np.zeros_like(S)
      sat_binary[(S > 90) & (S <= 100)] = 1

      hue_binary = np.zeros_like(H)
      hue_binary[(H > 0) & (H <= 10)] = 1


      binary_1 = cv2.bitwise_or(sx_binary, white_binary)
      binary_2 = cv2.bitwise_or(sat_binary, sat_binary)
      binary = cv2.bitwise_or(binary_1, binary_2)


      cv2.imshow("image", binary_2 * 255)


      with_lines = hist_detection(binary, warped)

      rev_warp = warp_frame(with_lines, True)

      final_result = resized_frame.copy()

      mask = np.zeros_like(final_result)
      mask[rev_warp > 0] = 255

      final_result[mask == 255] = rev_warp[mask == 255]

      cv2.imshow("result", final_result)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # Closing the Window
  cv2.destroyAllWindows()
  cap.release()


stream_processing()



