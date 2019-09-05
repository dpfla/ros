#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from lane_test.msg import msg_lane

def main():
	msg = msg_lane()sign
	sign = msg.sign



	if sign == "RIGHT SIGN":
		x1_ye = msg.x1_ye
		y1_ye = msg.y1_ye
		x2_ye = msg.x2_ye
		y2_ye = msg.y2_ye
		angle = np.pi - np.arctan((y1_ye - y2_ye) / (x1_ye - x2_ye)) * (180.0 / np.pi)
		msg.angle = angle


	if sign == "LEFT SIGN":
		x1_wh = msg.x1_wh
		y1_wh = msg.y1_wh
		x2_wh = msg.x2_wh
		y2_wh = msg.y2_wh
		angle = np.pi - np.arctan((y1_wh - y2_wh) / (x1_wh - x2_wh)) * (180.0 / np.pi)
		msg.angle = angle

	if sign == "DONT GO":
		if sign_1 == "RIGHT SIGN":
		    angle = np.pi - np.arctan((y1_ye - y2_ye) / (x1_ye - x2_ye)) * (180.0 / np.pi)
		    msg.angle = angle

		if sign_1 == "LEFT SIGN":
		    angle = np.pi - np.arctan((y1_wh - y2_wh) / (x1_wh - x2_wh)) * (180.0 / np.pi)
		    msg.angle = angle

if sign is not None:
	sign_1 = sign



