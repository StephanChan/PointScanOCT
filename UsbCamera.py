# -*- coding: utf-8 -*-
"""USB camera helpers for ASOCT live view."""

try:
    import cv2
except Exception:
    cv2 = None


USB_PIXEL_SIZE_MM = 0.045
USB_X_DISPLACEMENT_MM = -4.6
USB_Y_DISPLACEMENT_MM = 20.9
USB_CAMERA_INDEX = 0
USB_FRAME_WIDTH = 3840
USB_FRAME_HEIGHT = 2160
USB_AUTO_EXPOSURE = 0.25
USB_EXPOSURE = -5.0


def stage_to_usb_image(stage_x, stage_y, image_width):
    px = image_width - ((stage_y - USB_Y_DISPLACEMENT_MM) / USB_PIXEL_SIZE_MM)
    py = (stage_x - USB_X_DISPLACEMENT_MM) / USB_PIXEL_SIZE_MM
    return px, py


def orient_usb_frame(frame):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for USB camera frame orientation.")
    return cv2.flip(frame, 1)


def open_usb_camera(configure_exposure=False):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for USB camera live view.")
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_FRAME_HEIGHT)
    if configure_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, USB_AUTO_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, USB_EXPOSURE)
    return cap
