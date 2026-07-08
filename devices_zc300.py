import ctypes


class ZC300:
    def __init__(self) -> None:
        # 
        self.zc300 = ctypes.cdll.LoadLibrary(r"D:\LineScanOCT\LineScanOCT\zolix\zolix_zc_300.dll")

    def zc300_enum_count(self):
        """"""
        enum_dev = self.zc300.zc300_enum_count()
        return enum_dev

    def zc300_enum_info(self, index, buff_size):
        """
        index: ,0  ~  zc300_enum_count - 1.
                buff: , 16 .
                buff_size: """
        self.zc300.zc300_enum_info.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.zc300.zc300_enum_info.restype = ctypes.c_void_p

        a = ctypes.c_int(index)
        str_info = (ctypes.c_char*buff_size)()
        buff_size1 = ctypes.c_int(buff_size)

        self.zc300.zc300_enum_info(a, str_info, buff_size1)
        return str_info.value

    def zc300_open(self, dev_sn, addr):
        """zc300,zc300
        
                info: .
                addr: ,  1 ~ 255."""
        self.zc300.zc300_open.argtypes = [ctypes.c_char_p, ctypes.c_ubyte]
        self.zc300.zc300_open.restype = ctypes.c_bool

        # dev_sn_1 = ctypes.c_char_p(dev_sn)
        addr1 = ctypes.c_ubyte(addr)

        issuccess = self.zc300.zc300_open(dev_sn, addr1)
        # print('',dev_sn)
        return issuccess

    def zc300_close(self):
        """zc300"""
        self.zc300.zc300_close()
        return None

###########################################
    def zc300_get_sn(self):
        """
        zc300,"""
        self.zc300.zc300_get_sn.argtypes = [
            ctypes.POINTER(ctypes.c_long)]
        self.zc300.zc300_get_sn.restype = ctypes.c_bool

        sn1 = ctypes.c_long(999)
        self.zc300.zc300_get_sn(ctypes.byref(sn1))

        return sn1.value

    def zc300_get_model(self, buff_size):
        """
        zc300
        
                buff: , 16 .
                buff_size: .
        """
        self.zc300.zc300_get_model.argtypes = [
            ctypes.c_void_p, ctypes.c_int]
        self.zc300.zc300_get_model.restype = ctypes.c_bool

        buff1 = (ctypes.c_char*buff_size)()
        buff_size1 = ctypes.c_int(buff_size)
        self.zc300.zc300_get_model(
            buff1, buff_size1)

        return buff1.value

    def zc300_error_info(self, buff_size):
        """
        (),
        
                buff: , 64 .
                buff_size: .
        """
        self.zc300.zc300_error_info.argtypes = [
            ctypes.c_void_p, ctypes.c_int]
        self.zc300.zc300_error_info.restype = ctypes.c_bool

        buff1 = (ctypes.c_char*buff_size)()
        buff_size1 = ctypes.c_int(buff_size)
        self.zc300.zc300_error_info(
            buff1, buff_size1)

        return buff1.value

###########################################

    def zc300_set_enabled(self, axis, enabled):
        """
        
        
                axis: .
                enabled: true - , false - ."""
        self.zc300.zc300_set_enabled.argtypes = [
            ctypes.c_short, ctypes.c_bool]
        self.zc300.zc300_set_enabled.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        enabled1 = ctypes.c_bool(enabled)
        

        issuccess = self.zc300.zc300_set_enabled(axis, enabled1)
        return issuccess

    def zc300_get_enabled(self,  axis):
        """
        , .
        
                axis: .
                enabled: true - , false - ."""
        self.zc300.zc300_get_enabled.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_bool)]
        self.zc300.zc300_get_enabled.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        enabled1 = ctypes.c_bool(0)

        self.zc300.zc300_get_enabled(axis1, ctypes.byref(enabled1))
        return enabled1.value
###########################################

    def zc300_set_stage_type(self, axis, type1):
        """
        ,
        
                axis: 
                type: ,"""
        self.zc300.zc300_set_stage_type.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_set_stage_type.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        type11 = ctypes.c_uint(type1)

        issuccess = self.zc300.zc300_set_stage_type(axis, type1)
        return issuccess

    def zc300_get_stage_type(self, axis):
        """
        
        
                axis: 
                type: ,"""
        self.zc300.zc300_get_stage_type.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_stage_type.restype = ctypes.c_bool

        axis1 = ctypes.c_ubyte(axis)
        type1 = ctypes.c_short(5)

        self.zc300.zc300_get_stage_type(axis1, ctypes.byref(type1))
        return type1.value


###########################################

    def zc300_set_io_output(self, output):
        """
        IO, 0 ~ 8.
        
                output: IO """
        self.zc300.zc300_set_io_output.argtypes = [
            ctypes.c_short]
        self.zc300.zc300_set_io_output.restype = ctypes.c_bool

        output1 = ctypes.c_short(output)

        issuccess = self.zc300.zc300_set_io_output(output1)
        return issuccess

    def zc300_get_io_output(self):
        """
        IO.
        
                output: io."""
        self.zc300.zc300_get_io_output.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_io_output.restype = ctypes.c_bool

        output1 = ctypes.c_short(55)

        self.zc300.zc300_get_io_output(ctypes.byref(output1))
        return output1.value

    def zc300_get_io_input(self):
        """
        IO
        
                input: io"""
        self.zc300.zc300_get_io_input.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_io_input.restype = ctypes.c_bool

        input1 = ctypes.c_short(55)

        self.zc300.zc300_get_io_input(ctypes.byref(input1))
        return input1.value

###########################################
    def zc300_set_unit(self, axis, unit):
        """
         
            axis:  [0:X, 1:Y, 2:Z]
            unit:  [0:mm, 1:um, 2:pp, 3:deg]"""
        self.zc300.zc300_set_unit.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_set_unit.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        unit1 = ctypes.c_short(unit)

        issuccess = self.zc300.zc300_set_unit(axis1, unit1)
        return issuccess

    def zc300_get_unit(self, axis):
        """
         
            axis:  [0:X, 1:Y, 2:Z]
            unit:  [0:mm, 1:um, 2:pp, 3:deg]"""
        self.zc300.zc300_get_unit.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_unit.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        unit1 = ctypes.c_short(55)

        self.zc300.zc300_get_unit(axis1, ctypes.byref(unit1))
        return unit1.value


###########################################


    def zc300_set_pitch(self, axis, pitch):
        """
        -,:mm/r,:0 <pitch < 1000
        axis:  [0:X, 1:Y, 2:Z]
        pitch: """
        self.zc300.zc300_set_pitch.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_pitch.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        pitch1 = ctypes.c_float(pitch)

        issuccess = self.zc300.zc300_set_pitch(axis1, pitch1)
        return issuccess

    def zc300_get_pitch(self, axis):
        """
        -,:mm/r
        axis:  [0:X, 1:Y, 2:Z]
        pitch: """
        self.zc300.zc300_get_pitch.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_pitch.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        pitch1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_pitch(axis1, ctypes.byref(pitch1))
        return pitch1.value


###########################################


    def zc300_set_spr(self, axis, spr):
        """
        -,:pp/r,:0 <spr < 10000000.
        axis:  [0:X, 1:Y, 2:Z]
        spr: ,"""
        self.zc300.zc300_set_spr.argtypes = [
            ctypes.c_short, ctypes.c_long]
        self.zc300.zc300_set_spr.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        spr1 = ctypes.c_long(spr)

        issuccess = self.zc300.zc300_set_spr(axis1, spr1)
        return issuccess

    def zc300_get_spr(self, axis):
        """
        -,:pp/r
        axis:  [0:X, 1:Y, 2:Z]
        spr: ,"""
        self.zc300.zc300_get_spr.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_long)]
        self.zc300.zc300_get_spr.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        spr1 = ctypes.c_long(99)

        self.zc300.zc300_get_spr(axis1, ctypes.byref(spr1))
        return spr1.value


###########################################


    def zc300_set_ratio(self, axis, ratio):
        """
        -,:0 <ratio < 10000.
        axis:  [0:X, 1:Y, 2:Z]
        axis: .
                ratio: ,"""
        self.zc300.zc300_set_ratio.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_ratio.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        ratio1 = ctypes.c_float(ratio)

        issuccess = self.zc300.zc300_set_ratio(axis1, ratio1)
        return issuccess

    def zc300_get_ratio(self, axis):
        """-
        axis:  [0:X, 1:Y, 2:Z]
        axis: .
                ratio: ,"""
        self.zc300.zc300_get_ratio.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_ratio.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        ratio1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_ratio(axis1, ctypes.byref(ratio1))
        return ratio1.value

###########################################
    def zc300_set_init_speed(self, axis, speed):
        """
        ,, .
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_set_init_speed.argtypes = [
            ctypes.c_short,  ctypes.c_float]
        self.zc300.zc300_set_init_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_init_speed(axis1, speed1)
        return issuccess

    def zc300_get_init_speed(self, axis):
        """
        ,.
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_get_init_speed.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_init_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_init_speed(axis1, ctypes.byref(speed1))
        return speed1.value

###########################################
    def zc300_set_move_speed(self, axis, speed):
        """
        ,
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_set_move_speed.argtypes = [
            ctypes.c_short,  ctypes.c_float]
        self.zc300.zc300_set_move_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_move_speed(axis1, speed1)
        return issuccess

    def zc300_get_move_speed(self, axis):
        """
        ,
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_get_move_speed.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_move_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_move_speed(axis1, ctypes.byref(speed1))
        return speed1.value


###########################################


    def zc300_set_acc_speed(self, axis, speed):
        """
        
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_set_acc_speed.argtypes = [
            ctypes.c_short,  ctypes.c_float]
        self.zc300.zc300_set_acc_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_acc_speed(axis1, speed1)
        return issuccess

    def zc300_get_acc_speed(self, axis):
        """
        
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_get_acc_speed.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_acc_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_acc_speed(axis1, ctypes.byref(speed1))
        return speed1.value


###########################################


    def zc300_set_home_speed(self, axis, speed):
        """
        (),
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_set_home_speed.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_home_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(speed)

        issuccess = self.zc300.zc300_set_home_speed(axis1, speed1)
        return issuccess

    def zc300_get_home_speed(self, axis):
        """
        (),
        axis:  [0:X, 1:Y, 2:Z]
        speed: """
        self.zc300.zc300_get_home_speed.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_home_speed.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        speed1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_home_speed(axis1, ctypes.byref(speed1))
        return speed1.value


###########################################


    def zc300_set_home_mode(self, axis, mode):
        """
        
        axis: .
                mode: , ."""
        self.zc300.zc300_set_home_mode.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_set_home_mode.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        mode1 = ctypes.c_short(mode)

        issuccess = self.zc300.zc300_set_home_mode(axis1, mode1)
        return issuccess

    def zc300_get_home_mode(self, axis,mode =ctypes.c_short(55) ):
        """
        
        axis: .
                mode: , ."""
        self.zc300.zc300_get_home_mode.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_home_mode.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        mode1 = mode

        self.zc300.zc300_get_home_mode(axis1, ctypes.byref(mode1))
        return mode1.value

###########################################
    def zc300_move(self, axis, type1, distance):
        """
        ,.
        axis: .
        type: , .
        distance: ,,"""
        self.zc300.zc300_move.argtypes = [
            ctypes.c_short, ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_move.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        type11 = ctypes.c_short(type1)
        distance1 = ctypes.c_float(distance)

        issuccess = self.zc300.zc300_move(axis1, type11, distance1)
        return issuccess

###########################################

    def zc300_stop(self, axis, mode):
        """
        
        axis:  [0:X, 1:Y, 2:Z]
        mode:  [0:, 1:]"""
        self.zc300.zc300_stop.argtypes = [
            ctypes.c_short, ctypes.c_short]
        self.zc300.zc300_stop.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        mode1 = ctypes.c_short(mode)

        issuccess = self.zc300.zc300_stop(axis1, mode1)
        return issuccess


###########################################


    def zc300_home(self, axis):
        """
        """
        self.zc300.zc300_home.argtypes = [
            ctypes.c_short]
        self.zc300.zc300_home.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)

        issuccess = self.zc300.zc300_home(axis1)
        return issuccess

###########################################
    def zc300_set_position(self, axis, pos):
        """
        (),,:5,zc300_set_position 0,. 
        axis: .
		pos: ."""
        self.zc300.zc300_set_position.argtypes = [
            ctypes.c_short, ctypes.c_float]
        self.zc300.zc300_set_position.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        pos1 = ctypes.c_float(pos)

        issuccess = self.zc300.zc300_set_position(axis1, pos1)
        return issuccess

    def zc300_get_position(self, axis):
        """
        ,
        axis: .
		pos: """
        self.zc300.zc300_get_position.argtypes = [
            ctypes.c_short,  ctypes.POINTER(ctypes.c_float)]
        self.zc300.zc300_get_position.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        position1 = ctypes.c_float(9.9)

        self.zc300.zc300_get_position(axis1, ctypes.byref(position1))
        return position1.value

###########################################
    def zc300_get_idle(self, axis):
        """
        
        axis: .
		idle: ."""
        self.zc300.zc300_get_idle.argtypes = [
            ctypes.c_short, ctypes.POINTER(ctypes.c_bool)]
        self.zc300.zc300_get_idle.restype = ctypes.c_bool

        axis1 = ctypes.c_short(axis)
        idle1 = ctypes.c_bool(1)

        self.zc300.zc300_get_idle(axis1, ctypes.byref(idle1))
        return idle1.value

###########################################
    def zc300_get_status(self):
        """
        ,:,,
        status: , """
        self.zc300.zc300_get_status.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_status.restype = ctypes.c_bool

        status1 = ctypes.c_short(99)

        self.zc300.zc300_get_status(ctypes.byref(status1))
        return status1.value

###########################################
    def zc300_set_buzzer(self, status):
        """
        
        status: ,"""
        self.zc300.zc300_set_buzzer.argtypes = [
            ctypes.c_short]
        self.zc300.zc300_set_buzzer.restype = ctypes.c_bool

        status1 = ctypes.c_short(status)
        issuccess = self.zc300.zc300_set_buzzer(status1)
        return issuccess

    def zc300_get_buzzer(self):
        """
        
        status: """
        self.zc300.zc300_get_buzzer.argtypes = [
            ctypes.POINTER(ctypes.c_short)]
        self.zc300.zc300_get_buzzer.restype = ctypes.c_bool

        status1 = ctypes.c_short(55)

        self.zc300.zc300_get_buzzer(ctypes.byref(status1))
        return status1.value
