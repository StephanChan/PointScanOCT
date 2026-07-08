# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:32:29 2026

@author: shuaibin
"""

"""
 ZC300 
zc300_set_spr  1600 200*1600
"""

import time
from devices_zc300 import ZC300  #  SDK 
# XYZ ,X Y Z
# mm
PITCH = [5,4,10]
# 
REDUCTION = [1,1,120]

#  SDK 
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

UNIT_PP = 0      # 
UNIT_MM = 1      # 
UNIT_DEG = 2     # 

STAGE_LINEAR = 0
STAGE_ROTARY = 1

HOME_BY_USER = 1
HOME_BY_LIMIT = 2
HOME_BY_ZERO = 3

MOVE_ABSOLUTE = 0
MOVE_RELATIVE = 1
MOVE_CONTINUOUS = 2

STOP_SLOWLY = 0
STOP_IMMEDIATELY = 1



class ZC300MotorController:
    """ZC300  spr """

    def __init__(self, port_info=None, addr=1, auto_connect=True):
        """
        
        :param port_info:  'COM3'  None
        :param addr: 1~255
        :param auto_connect: 
        """
        self.zc = ZC300()
        self.addr = addr
        self.connected = False

        if auto_connect:
            self.connect(port_info)

        # 
        self.axis_config = {
            AXIS_X: {},
            AXIS_Y: {},
            AXIS_Z: {}
        }

    def connect(self, port_info=None):
        """"""
        if port_info is None:
            dev_count = self.zc.zc300_enum_count()
            if dev_count == 0:
                raise RuntimeError(" ZC300 ")
            buff_size = 64
            port_info = self.zc.zc300_enum_info(0, buff_size).decode('ascii', errors='ignore')
            print(f": {port_info}")

        ret = self.zc.zc300_open(port_info.encode(), self.addr)
        if not ret:
            raise RuntimeError(f": {port_info}, : {self.addr}")
        self.connected = True
        print("ZC300 ")

    def close(self):
        """"""
        if self.connected:
            self.zc.zc300_close()
            self.connected = False
            print("ZC300 ")

    def __del__(self):
        self.close()

    # ====================  ====================
    def configure_axis(self, axis, subdivide=1600,
                       unit=UNIT_MM, stage_type=STAGE_LINEAR,
                       direction_reverse=True):
        """
        
        :param axis:  (AXIS_X/Y/Z)
        :param lead_screw:  (mm/r)
        :param reduction:  ( / ) 1.0
        :param subdivide: 1.8  subdivide  1600
        :param unit:  (UNIT_MM / UNIT_PP / UNIT_DEG)
        :param stage_type:  (STAGE_LINEAR / STAGE_ROTARY)
        :param direction_reverse: 
        """
        if not self.connected:
            raise RuntimeError("")

        #  =  /  DLL 
        effective_pitch = PITCH[axis] / REDUCTION[axis]

        # zc300_set_spr  1600 200*1600
        #  SDK  spr 1.8
        spr = subdivide

        #  SDK 
        self.zc.zc300_set_stage_type(axis, stage_type)
        self.zc.zc300_set_unit(axis, unit)
        self.zc.zc300_set_pitch(axis, effective_pitch)
        self.zc.zc300_set_spr(axis, spr)   # 
        self.zc.zc300_set_home_mode(axis, HOME_BY_ZERO)
        #  ratio  DLL 

        # 
        self.axis_config[axis] = {
            'effective_pitch': effective_pitch,
            'spr': spr,
            'unit': unit,
            'direction_reverse': direction_reverse,
            'reduction': REDUCTION[axis],
            'lead_screw': PITCH[axis],
            'subdivide': subdivide
        }
        print(f"{axis} : ={effective_pitch:.4f} mm/r, ={spr}")

    # ====================  ====================
    def set_enable(self, axis, enable):
        """ (True=, False=)"""
        return self.zc.zc300_set_enabled(axis, enable)

    def get_enable(self, axis):
        """"""
        return self.zc.zc300_get_enabled(axis)

    # ==================== / ====================
    def set_init_speed(self, axis, speed_mm_s):
        """ (: mm/s)"""
        return self.zc.zc300_set_init_speed(axis, speed_mm_s)

    def set_move_speed(self, axis, speed_mm_s):
        """ (: mm/s)"""
        return self.zc.zc300_set_move_speed(axis, speed_mm_s)

    def set_acceleration(self, axis, acc_mm_s2):
        """ (: mm/s)"""
        return self.zc.zc300_set_acc_speed(axis, acc_mm_s2)

    def set_home_speed(self, axis, speed_mm_s):
        """ (: mm/s)"""
        return self.zc.zc300_set_home_speed(axis, speed_mm_s)

    def set_home_mode(self, axis, mode):
        """
        
        :param mode: HOME_BY_USER, HOME_BY_LIMIT, HOME_BY_ZERO
        """
        return self.zc.zc300_set_home_mode(axis, mode)

    # ====================  ====================
    def _apply_direction(self, axis, distance):
        """"""
        if self.axis_config[axis].get('direction_reverse', False):
            return -distance
        return distance

    def move_relative(self, axis, distance_mm):
        """
        
        :param distance_mm:  (mm)
        """
        dist = self._apply_direction(axis, distance_mm)
        result = self.zc.zc300_move(axis, MOVE_RELATIVE, dist)
        while not self.is_idle(axis):
            time.sleep(0.2)
        return result

    def move_absolute(self, axis, position_mm):
        """
        
        :param position_mm:  (mm)
        """
        #  direction_reverse 
        result = self.zc.zc300_move(axis, MOVE_ABSOLUTE, position_mm)
        while not self.is_idle(axis):
            time.sleep(0.2)
        return result

    def move_continuous(self, axis, direction_positive=True):
        """
        
        :param direction_positive: True=, False=
        """
        distance = 1.0 if direction_positive else -1.0
        dist = self._apply_direction(axis, distance)
        result = self.zc.zc300_move(axis, MOVE_CONTINUOUS, dist)
        while not self.is_idle(axis):
            time.sleep(0.2)
        return result

    def stop(self, axis, immediate=False):
        """
        
        :param immediate: True=, False=
        """
        mode = STOP_IMMEDIATELY if immediate else STOP_SLOWLY
        return self.zc.zc300_stop(axis, mode)

    def home(self, axis):
        """
        
        :param wait: 
        :param timeout_s: 
        """
        # self.move_absolute(axis, -1)
        ret = self.zc.zc300_home(axis)
        if not ret:
            return False
        while not self.is_idle(axis):
            time.sleep(0.2)
        # time.sleep(2)
        # while not self.is_idle(axis):
        #     time.sleep(0.1)
        # print(self.is_idle(axis))
        self.set_position(axis, 0)
        return True

    # ====================  ====================
    def get_position(self, axis):
        """ (: mm)"""
        return self.zc.zc300_get_position(axis)

    def set_position(self, axis, pos_mm):
        """"""
        return self.zc.zc300_set_position(axis, pos_mm)

    def is_idle(self, axis):
        """"""
        return self.zc.zc300_get_idle(axis)

    def get_status(self):
        """"""
        return self.zc.zc300_get_status()

    def get_axis_limit_status(self, axis):
        """
        /
        : {'positive_limit': bool, 'negative_limit': bool, 'origin': bool}
        """
        status = self.get_status()
        if axis == AXIS_X:
            pos_limit = bool(status & 0x0001)
            neg_limit = bool(status & 0x0002)
            origin = bool(status & 0x0004)
        elif axis == AXIS_Y:
            pos_limit = bool(status & 0x0008)
            neg_limit = bool(status & 0x0010)
            origin = bool(status & 0x0020)
        elif axis == AXIS_Z:
            pos_limit = bool(status & 0x0040)
            neg_limit = bool(status & 0x0080)
            origin = bool(status & 0x0100)
        else:
            return {}
        return {
            'positive_limit': pos_limit,
            'negative_limit': neg_limit,
            'origin': origin
        }

    def wait_idle(self, axis, timeout_s=10):
        """"""
        start = time.time()
        while not self.is_idle(axis):
            if time.time() - start > timeout_s:
                return False
            time.sleep(0.2)
        return True


# ====================  ====================
if __name__ == "__main__":
    # 
    motor = ZC300MotorController()
    AXIS = AXIS_X
    try:
        #  X 4mm1600
        motor.configure_axis(AXIS)   # 

        # 
        motor.set_init_speed(AXIS, 0.1)      #  1 mm/s
        motor.set_move_speed(AXIS, 10)     #  20 mm/s
        motor.set_acceleration(AXIS, 10)   #  50 mm/s
        motor.set_home_speed(AXIS, 10)      #  5 mm/s
        # motor.set_home_mode(AXIS, HOME_BY_ZERO)  # 

        # 
        motor.set_enable(AXIS, True)

        # 
        print("...")
        motor.home(AXIS)
        print(f": {motor.get_position(AXIS)} mm")

        #  10mm
        motor.move_relative(AXIS, 1.0)
        print(f": {motor.get_position(AXIS)} mm")

        #  5mm 
        # motor.move_absolute(AXIS_X, 5.0)
        # motor.wait_idle(AXIS_X)
        # print(f": {motor.get_position(AXIS_X)} mm")

        # 
        motor.set_enable(AXIS, False)

    finally:
        motor.close()
