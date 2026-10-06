# -*- coding: utf-8 -*-
"""Smoke test for an I-TEK IKap CameraLink line-scan camera."""

import os
import sys


IKAP_ROOT = r"C:\Program Files\I-TEK OptoElectronics\IKapLibrary"
IKAP_EXAMPLES_PYTHON = os.path.join(IKAP_ROOT, "Examples", "Python")
IKAP_LIB_DIR = os.path.join(IKAP_EXAMPLES_PYTHON, "IKapLib")
IKAP_DLL_DIR = os.path.join(IKAP_ROOT, "Dll", "C", "x64")
IKAP_GENICAM_DIR = os.path.join(IKAP_ROOT, "GenICam", "bin", "Win64_x64")


def bootstrap_ikap():
    for dll_dir in (IKAP_DLL_DIR, IKAP_GENICAM_DIR):
        if os.path.isdir(dll_dir):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_dir)
    if IKAP_LIB_DIR not in sys.path:
        sys.path.append(IKAP_LIB_DIR)


def decode_bytes(value):
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def smoke_test_camera():
    bootstrap_ikap()
    import IKapBoard
    import IKapC
    import IKapCDef

    res = IKapC.ItkManInitialize()
    if res != IKapCDef.ITKSTATUS_OK:
        print("ItkManInitialize failed: %#x" % res)
        return 1

    try:
        res, board_count = IKapBoard.IKapGetBoardCount(IKapBoard.IKBoardPCIE)
        if res != IKapBoard.IK_RTN_OK:
            print("IKapGetBoardCount failed: %#x" % res)
            return 1
        print("IKap PCIe board count:", board_count)
        for index in range(board_count):
            res, name = IKapBoard.IKapGetBoardName(IKapBoard.IKBoardPCIE, index)
            if res == IKapBoard.IK_RTN_OK:
                print("  board[%d]: %s" % (index, decode_bytes(name)))

        res, camera_count = IKapC.ItkManGetDeviceCount()
        if res != IKapCDef.ITKSTATUS_OK:
            print("ItkManGetDeviceCount failed: %#x" % res)
            return 1
        print("IKap camera count:", camera_count)
        if camera_count == 0:
            return 1

        opened = False
        for index in range(camera_count):
            res, info = IKapC.ItkManGetDeviceInfo(index)
            if res != IKapCDef.ITKSTATUS_OK:
                continue
            print(
                "  camera[%d]: name=%s, serial=%s, class=%s"
                % (
                    index,
                    decode_bytes(info.FullName),
                    decode_bytes(info.SerialNumber),
                    decode_bytes(info.DeviceClass),
                )
            )
            if info.DeviceClass == b"CameraLink" and info.SerialNumber != b"" and not opened:
                res, h_camera = IKapC.ItkDevOpen(
                    index,
                    IKapCDef.ITKDEV_VAL_ACCESS_MODE_EXCLUSIVE,
                )
                if res != IKapCDef.ITKSTATUS_OK or h_camera.value == 0:
                    print("Open CameraLink camera[%d] failed: %#x" % (index, res))
                    continue
                print("Opened CameraLink camera[%d] successfully." % index)
                IKapC.ItkDevClose(h_camera)
                opened = True

        return 0 if opened else 1
    finally:
        IKapC.ItkManTerminate()


if __name__ == "__main__":
    raise SystemExit(smoke_test_camera())
