#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Image:

    def __init__(self):
        self.bridge = CvBridge()
        self.count = 0
        self.avr_gradient = 0
        self.gradient = 0
        self.Threshold_up = 150
        self.Threshold_low = 100
        self.img_sub = 0
        self.img_pub = 0
        self.img = 0
        self.img_rst = 0
        self.gray_img = 0
        self.edge = 0
        self.line = 0

	def sub_img(self, image):
		self.img_sub = self.bridge.imgmsg_to_cv2(image)
	
	def pub_img(self, image):
		self.img_pub = self.bridge.cv2_to_imgmsg(image)

	def frame (self, image):
		self.img = image
		self.img_rst = self.img

	def bgr_to_gray(self):
		self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)


	def edge(self):
		self.Threshold_up = 150
		self.Threshold_low = 100
		self.edge = cv2.Canny(self.gray_img, self.Threshold_up, self.Threshold_low, 10)


	def get_line(self):
		self.line = cv2.HoughLinesP(self.edge, rho=1, theta=1 * np.pi / 180, threshold=8, minLineLength=10,
									maxLineGap=50)


	def get_gradient(self):
		l = self.line
		for i in len.l:
			L = self.line[i]

			if not (L[2] - L[0]):
				gradient = (L[3] - L[1]) / (L[2] - L[0])
			else:
				gradient = 99

			if gradient != 0 and L[2] == 199 and (self.gray_img.at(L[1], L[0]) > 240 or self.gray_img.at(L[3], L[2]) > 240):
				cv2.line(self.img_rst, (L[0], L[1]), (L[2], L[3]), (0, 0, 255), 2)
				cv2.circle(self.img_rst, (L[0], L[1]), 3, (0, 255, 0), 2, 3)
				cv2.circle(self.img_rst, (L[2], L[3]), 3, (0, 255, 0), 2, 3)
				self.count += 1
				self.avr_gradient += self.gradient
				cv2.imshow("a", self.img)

def main() :
	pub = rospy.Publisher('/image_raw', Image, queue_size=1)
	rospy.init_node('webcam', anonymous=True)
	rate = rospy.Rate(10)
	img = Image
	cap = cv2.VideoCapture(0)
	if (cap.isOpened() == False): 
		print("Unable to read camera feed")
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))

	while not rospy.is_shutdown():
		ret, frame = cap.read()
		if ret:
			img.frame(frame)
			img.bgr_to_gray()
			img.edge()
			img.get_line()
			img.get_gradient()
			img.pub_img(img.img_rst)
			pub.publish(img.img_pub)
			cv2.imshow(img.img_rst)

		rate.sleep()
