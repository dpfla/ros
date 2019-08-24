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
        self.img_rst = self.ori_img[240:480, 0:640]
        self.img_ye = self.ori_img[100:240, 0:320]
        self.img_wh = self.ori_img[100:240, 320:640]
        self.img_copy = self.img_rst.copy()

    def bgr_to_gray(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def wh_mask(self):
        change_hls = cv2.cvtColor(self.img_wh, cv2.COLOR_BGR2HLS)
        lower_wh = np.array([30, 140, 0])
        upper_wh = np.array([179, 255, 150])
        on_v = cv2.inRange(change_hls, lower_wh, upper_wh)
        # self.mask_wh = cv2.threshold(self.img_wh, 140, 255, cv2.THRESH_BINARY_INV)
        self.img_wh = cv2.bitwise_and(self.img_wh, self.img_wh, mask=on_v)
        # image = cv2.cvtColor(self.img_wh, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.GaussianBlur(image, (5, 5), 0)
        # self.img_wh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)

    def ye_mask(self):
        change_hsv = cv2.cvtColor(self.img_ye, cv2.COLOR_BGR2HSV)
        lower_ye = np.array([20, 100, 100])
        upper_ye = np.array([50, 255, 255])
        on_v = cv2.inRange(change_hsv, lower_ye, upper_ye)
        self.mask_ye = cv2.threshold(self.img_ye, 140, 255, cv2.THRESH_BINARY_INV)
        self.img_ye = cv2.bitwise_and(self.img_ye, self.img_wh, mask=on_v)

    def img_hstack(self, img1, img2, img3):
        img1 = np.hstack([img2, img3])

    def edge(self, img):
        self.bgr_to_gray(img)
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        self.img_canny = cv2.Canny(blur, 150, 200)

    def filter_region(self, img, vertices):
        mask = np.zeros_like(img)
        if len(mask.shape) == 2:
            cv2.fillPoly(mask, vertices, 255)
        else:
            cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
        return cv2.bitwise_and(img, mask)

    def select_region(self, img):
        rows, cols = img.shape[:2]
        bottom_left = [cols * 0.0, rows * 0.6]
        top_left = [cols * 0.0, rows * 0.4]
        bottom_right = [cols * 1, rows * 0.6]
        top_right = [cols * 0.85, rows * 0.4]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        self.img_canny = self.filter_region(img, vertices)

    def get_line(self):
        self.line = cv2.HoughLinesP(self.img_canny, rho=1, theta=np.pi / 180, threshold=50, minLineLength=1,
                                    maxLineGap=30)

        if self.line is not None:
            self.lane_mode = True

            for line in self.line:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.img_copy, (x1, y1), (x2, y2), [255, 0, 0], 2)
        else:
            self.lane_mode = False

    def average_slope_intercept(self):
        left_lines = []
        left_weights = []
        right_lines = []
        right_weights = []

        for line in self.line:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = round(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2), 3)
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))

        self.line_ye = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        self.line_wh = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(
            right_weights) > 0 else None

    def make_line_points_ye(self, y1, y2):
        if self.line_ye is None:
            return None

        slope, intercept = self.line_ye

        x1_ye = np.float(y1 - intercept) / slope
        x2_ye = np.float(y2 - intercept) / slope

        if math.isinf(x1_ye):
            x1_ye = 99

        if math.isinf(x2_ye):
            x2_ye = 99

        self.x1_ye = int(x1_ye)
        self.x2_ye = int(x2_ye)
        self.y1_ye = int(y1)
        self.y2_ye = int(y2)

    def make_line_points_wh(self, y1, y2):
        if self.line_wh is None:
            return None

        slope, intercept = self.line_wh

        x1_wh = np.float(y1 - intercept) / slope
        x2_wh = np.float(y2 - intercept) / slope

        if math.isinf(x1_wh):
            x1_wh = 99

        if math.isinf(x2_wh):
            x2_wh = 99

        self.x1_wh = int(x1_wh)
        self.x2_wh = int(x2_wh)
        self.y1_wh = int(y1)
        self.y2_wh = int(y2)

    def draw_lines(self):
        y1_1 = self.img_canny.shape[0]  # bottom of the image
        y2_1 = y1_1 * 0.6  # slightly lower than the middle

        pre_point_wh = (self.x1_wh, self.x2_wh, self.y1_wh, self.y2_wh)
        pre_point_ye = (self.x1_ye, self.x2_ye, self.y1_ye, self.y2_ye)

        self.make_line_points_ye(y1_1, y2_1)
        self.make_line_points_wh(y1_1, y2_1)

        if self.lane_mode is True:
            if self.x1_wh is None and self.x2_wh is None and self.y1_wh is None and self.y2_wh is None:
                self.x1_wh, self.x2_wh, self.y1_wh, self.y2_wh = pre_point_wh

            elif self.x1_ye is None and self.x2_ye is None and self.y1_ye is None and self.y2_ye is None:
                self.x1_ye, self.x2_ye, self.y1_ye, self.y2_ye = pre_point_ye

            if self.line_wh is not None and self.line_ye is None:
                if self.x1_wh == self.x2_wh:
                    self.angle = 180 * (180.0 / np.pi)
                else:
                    self.angle = np.pi - np.arctan((self.y1_wh - self.y2_wh) / (self.x1_wh - self.x2_wh)) * (180.0 / np.pi)
                cv2.line(self.img_copy, (self.x1_wh - 450, self.y1_wh), (self.x2_wh - 450, self.y2_wh), [0, 255, 0], 2)

            elif self.line_wh is None and self.line_ye is not None:
                if self.x1_ye == self.x2_ye:
                    self.angle = 180 * (180.0 / np.pi)
                else:
                    self.angle = np.pi - np.arctan((self.y1_ye - self.y2_ye) / (self.x1_ye - self.x2_ye)) * (180.0 / np.pi)
                cv2.line(self.img_copy, (self.x1_ye - 450, self.y1_ye), (self.x2_ye - 450, self.y2_ye), [0, 255, 0], 2)

            elif not (self.line_wh is None and self.line_ye is None):
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

            elif self.line_wh is None and self.line_ye is None :
                pass

            self.__init__()


# self.x1_wh == None and self.x1_ye == None and self.x2_wh == None and self.x2_ye == None and self.y1_wh == None and self.y1_ye == None and self.y2_wh= None and self.y2_ye== None


def main():
    img_now = Image()
    cap = cv2.VideoCapture('lane_true_4.avi')#(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("open fail video")
    while True:
        ret, frame = cap.read()
        if ret:
            img_now.frame_img(frame)
            img_now.edge(img_now.img_rst)
            # img_now.select_region(img_now.img_canny)
            img_now.get_line()
            if img_now.lane_mode == True:
                img_now.average_slope_intercept()
                img_now.draw_lines()

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
