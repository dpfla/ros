#!/usr/bin/env python
import numpy as np
import cv2
import math

class Image:
    def __init__(self):
        self.line = []
        self.line_md = []
        self.line_wh = []
        self.line_ye = []
        self.ye_pose = []
        self.wh_pose = []
        self.wh_true = False
        self.ye_true = False
        self.lane_mode = None
        self.x1_ye = None
        self.x2_ye = None
        self.y1_ye = None
        self.y2_ye = None
        self.x1_wh = None
        self.x2_wh = None
        self.y1_wh = None
        self.y2_wh = None

    def frame_img(self, image):
        self.ori_img = image
        self.img_rst = self.ori_img[300:480, 0:640]
        self.img_copy = self.img_rst.copy()

    def bgr_to_gray(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def wh_mask(self):
        img_rgb = self.img_rst
        img_hls = self.img_rst
        # change_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        lower_rgb_wh = np.array([245, 245, 245])
        upper_rgb_wh = np.array([255, 255, 255])
        on_v = cv2.inRange(img_rgb, lower_rgb_wh, upper_rgb_wh)
        img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=on_v)

        change_hls = cv2.cvtColor(img_hls, cv2.COLOR_BGR2HLS)
        lower_hls_wh = np.array([0, 250, 0])
        upper_hls_wh = np.array([179, 255, 255])
        on_v = cv2.inRange(change_hls, lower_hls_wh, upper_hls_wh)
        img_hls = cv2.bitwise_and(img_hls, img_hls, mask=on_v)

        self.img_wh_mask = cv2.bitwise_and(img_hls, img_rgb)


    def ye_mask(self):
        img_hsv = self.img_rst
        img_hls = self.img_rst

        change_hls = cv2.cvtColor(img_hls,  cv2.COLOR_BGR2HLS)
        lower_rgb_ye = np.array([20, 0, 150])
        upper_rgb_ye = np.array([40, 255, 255])
        on_v = cv2.inRange(change_hls, lower_rgb_ye, upper_rgb_ye)
        img_hls = cv2.bitwise_and(img_hls, img_hls, mask=on_v)

        change_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
        lower_ye = np.array([20, 100, 100])
        upper_ye = np.array([70, 255, 255])
        on_v = cv2.inRange(change_hsv, lower_ye, upper_ye)
        img_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=on_v)
        self.img_ye_mask = cv2.bitwise_and(img_hsv, img_hls)


    def edge(self):
        img = cv2.bitwise_or(self.img_wh_mask, self.img_ye_mask)
        self.img_mask = img
        self.bgr_to_gray(img)
        blur = cv2.GaussianBlur(img, (3, 3), 0.5)
        kernel = np.ones((3, 3), np.uint8)
        blur = cv2.dilate(blur, kernel, iterations=1)
        self.img_mask = blur
        self.img_canny = cv2.Canny(blur, 150, 200)

    def get_line(self):
        self.line = cv2.HoughLinesP(self.img_canny, rho=1, theta=np.pi / 180, threshold=50, minLineLength=1,
                                    maxLineGap=30)

        if self.line is not None:
            self.lane_mode = True
        else:
            self.lane_mode = False

    def average_slope_intercept(self):
        slope_max_wh = -1
        slope_max_ye = -1

        for line in self.line:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                slope_abs = abs(slope)

                if slope < 0:
                    if slope_max_ye < slope_abs:
                        slope_max_ye = slope_abs
                        self.ye_pose = [x1, y1, x2, y2]
                        #cv2.line(self.img_copy, (x1_ye, y1_ye), (x2_ye, y2_ye), [255, 0, 0], 2)

                else:
                    if slope_max_wh < slope:
                        slope_max_wh = slope_abs
                        self.wh_pose = [x1, y1, x2, y2]

        if not self.ye_pose == []:
            self.ye_true = True
            self.pre_point_wh = (self.x1_wh, self.x2_wh, self.y1_wh, self.y2_wh)
            self.x1_ye, self.y1_ye, self.x2_ye, self.y2_ye = self.ye_pose
            cv2.line(self.img_copy, (self.x1_ye, self.y1_ye), (self.x2_ye, self.y2_ye), [255, 0, 0], 2)

        if not self.wh_pose == []:
            self.wh_true = True
            self.pre_point_ye = (self.x1_ye, self.x2_ye, self.y1_ye, self.y2_ye)
            self.x1_wh, self.y1_wh, self.x2_wh, self.y2_wh = self.wh_pose
            cv2.line(self.img_copy, (self.x1_wh, self.y1_wh), (self.x2_wh, self.y2_wh), [0, 0, 255], 2)


    def draw_lines(self):
        if self.lane_mode is True:
            if self.x1_wh is None and self.x2_wh is None and self.y1_wh is None and self.y2_wh is None:
                self.x1_wh, self.x2_wh, self.y1_wh, self.y2_wh = self.pre_point_wh

            elif self.x1_ye is None and self.x2_ye is None and self.y1_ye is None and self.y2_ye is None:
                self.x1_ye, self.x2_ye, self.y1_ye, self.y2_ye = self.pre_point_ye

            if self.ye_true is False and self.wh_true is True:
                if self.x1_wh == self.x2_wh:
                    self.angle = 180 * (180.0 / np.pi)
                else:
                    self.angle = np.pi - np.arctan((self.y1_wh - self.y2_wh) / (self.x1_wh - self.x2_wh)) * (
                            180.0 / np.pi)
                cv2.line(self.img_copy, (self.x1_wh - 400, self.y1_wh), (self.x2_wh - 400, self.y2_wh), [0, 255, 0], 2)

            elif self.wh_true is False and self.ye_true is True:
                if self.x1_ye == self.x2_ye:
                    self.angle = 180 * (180.0 / np.pi)
                else:
                    self.angle = np.pi - np.arctan((self.y1_ye - self.y2_ye) / (self.x1_ye - self.x2_ye)) * (
                            180.0 / np.pi)
                cv2.line(self.img_copy, (self.x1_ye + 400, self.y1_ye), (self.x2_ye + 400, self.y2_ye), [0, 255, 0], 2)

            elif self.ye_true is True and self.wh_true is True:
                x1 = int((self.x1_wh + self.x1_ye) // 2)
                x2 = int((self.x2_wh + self.x2_ye) // 2)
                y1 = int((self.y1_wh + self.y1_ye) // 2)
                y2 = int((self.y2_wh + self.y2_ye) // 2)

                if x2 == x1:
                    self.angle = np.pi - np.arctan(y1 / x1) * (180.0 / np.pi)

                else:
                    self.angle = np.pi - np.arctan((y1 - y2) / (x1 - x2)) * (180.0 / np.pi)

                self.line_md.append((x1, y1, x2, y2))
                cv2.line(self.img_copy, (x1, y1), (x2, y2), [0, 255, 0], 2)

            elif self.ye_true is False and self.wh_true is False:
                pass

            self.__init__()


def main():
    img_now = Image()
    cap = cv2.VideoCapture('lane_1.avi')  # (0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("open fail video")
    while True:
        ret, frame = cap.read()

        if ret:
            img_now.frame_img(frame)
            img_now.wh_mask()
            img_now.ye_mask()
            img_now.edge()
            img_now.get_line()
            if img_now.lane_mode == True:
                img_now.average_slope_intercept()
                img_now.draw_lines()

            img_mask = cv2.bitwise_or(img_now.img_ye_mask, img_now.img_wh_mask)
            cv2.imshow("img", img_now.img_copy)

            k = cv2.waitKey(10)
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                break


try:
    main()
except TypeError as te:
    print("[TypeError]", te)
