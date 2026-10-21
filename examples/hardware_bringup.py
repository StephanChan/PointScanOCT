"""真机 bring-up：相机 → DAQ → 联合，三步分开验证（也可 ``--simulate`` 空跑同一套流程）。

用法示例：
    python examples/hardware_bringup.py                 # 三步全跑
    python examples/hardware_bringup.py --camera-only
    python examples/hardware_bringup.py --aodo-only --hold 3
    python examples/hardware_bringup.py --joint --save cscan.tif
    python examples/hardware_bringup.py --simulate      # 没硬件时验证脚本本身

每一步都会打印"PASS/FAIL + 关键观测值"，退出码 0 表示全部通过。
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import octsdk
except ModuleNotFoundError:  # 直接在仓库里跑示例时，把仓库根目录加进 sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import octsdk


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="OCT SDK 真机 bring-up")
    parser.add_argument("--camera-only", action="store_true", help="只测相机采集")
    parser.add_argument("--aodo-only", action="store_true", help="只测振镜 / DO 输出")
    parser.add_argument("--joint", action="store_true", help="B-scan + C-scan 全链路")
    parser.add_argument("--all", action="store_true", help="三步全跑（默认）")
    parser.add_argument("--simulate", action="store_true", help="用 SIM 相机/SIM AODO 空跑")
    parser.add_argument("--preset", default="12bit4tap", help="随包 .vlcf 预设名")
    parser.add_argument("--samples", type=int, default=1440, help="光谱采样数（应与 .vlcf IMAGE_WIDTH 一致）")
    parser.add_argument("--x-positions", type=int, default=200, help="每条 B-line 的振镜位置数")
    parser.add_argument("--x-step-um", type=float, default=10.0)
    parser.add_argument("--y-lines", type=int, default=8)
    parser.add_argument("--y-step-um", type=float, default=10.0)
    parser.add_argument("--flyback", type=int, default=300, help="回扫采样点数")
    parser.add_argument("--line-rate", type=float, default=10000.0, help="相机行频 Hz")
    parser.add_argument("--exposure", type=float, default=4.2, help="曝光 µs")
    parser.add_argument("--mm-per-volt", type=float, default=2.3776, help="振镜标定 mm/V")
    parser.add_argument("--depth-start", type=int, default=0, help="深度窗起点（裁掉 DC）")
    parser.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--packages", type=int, default=20, help="相机步骤最多收多少包")
    parser.add_argument("--timeout", type=float, default=60.0, help="每步最多等待秒数")
    parser.add_argument("--hold", type=float, default=None, help="DAQ 步保持输出秒数（默认=扫描时长）")
    parser.add_argument("--save", default=None, help="联合步骤的 C-scan TIFF 路径")
    parser.add_argument("--ikap-root", default=None, help="IKap SDK 安装目录")
    parser.add_argument("--artdaq-lib", default=None, help="ART-DAQ Python 库目录")
    return parser.parse_args(argv)


def build_config(args):
    simulate = bool(args.simulate)
    return octsdk.OctConfig(
        scan=octsdk.ScanConfig(
            x_step_um=args.x_step_um,
            x_length_um=args.x_step_um * args.x_positions,
            y_step_um=args.y_step_um,
            y_lines=args.y_lines,
            flyback_samples=args.flyback,
        ),
        processing=octsdk.ProcessingConfig(
            raw_spectral_samples=args.samples,
            aline_avg=1,
            bline_avg=1,
            depth_start=args.depth_start,
            amplification=1.0,
        ),
        camera=octsdk.CameraConfig(
            preset=args.preset,
            simulate=simulate,
            line_rate_hz=args.line_rate,
            exposure_us=args.exposure,
            sdk_root=args.ikap_root,
        ),
        aodo=octsdk.AodoConfig(
            simulate=simulate,
            x_mm_per_volt=args.mm_per_volt,
            y_mm_per_volt=args.mm_per_volt,
            sdk_root=args.artdaq_lib,
        ),
        engine=octsdk.EngineConfig(package_y_lines=max(1, min(64, args.y_lines))),
    )


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def step_environment(config):
    """打印运行环境与 .vlcf 摘要（不判定成败，只用于对照）。"""
    report_data = octsdk.environment_report()
    print(
        "环境: python {python} | numpy {numpy} | cupy {cupy} | 设备 {devices}".format(
            python=report_data["python"],
            numpy=report_data["numpy"].get("version", "?"),
            cupy=report_data["cupy"].get("version", "缺失"),
            devices=[device["name"] for device in report_data["cupy"].get("devices", [])],
        )
    )
    print(
        f"环境: IKap={report_data['ikap_sdk']['exists']} ({report_data['ikap_sdk']['path']}) | "
        f"ART-DAQ={report_data['artdaq_sdk']['exists']} ({report_data['artdaq_sdk']['path']})"
    )
    try:
        vlcf_path = octsdk.camera.resolve_vlcf_path(config.camera)
        summary = octsdk.camera.describe_vlcf(
            octsdk.camera.read_vlcf_settings(vlcf_path)
        )
        print(
            f"配置: {vlcf_path.name} depth={summary['pixel_depth']} taps={summary['taps']} "
            f"width={summary['image_width']} offset_x={summary['offset_x']} "
            f"line_trigger={summary['trigger_source']}"
        )
        if summary["image_width"] and int(summary["image_width"]) != int(
            config.processing.raw_spectral_samples
        ):
            print(
                f"  注意: .vlcf IMAGE_WIDTH={summary['image_width']} 与 --samples="
                f"{config.processing.raw_spectral_samples} 不一致（板卡以 .vlcf 为准）"
            )
    except Exception as error:
        print("配置: .vlcf 解析失败 —", error)


def step_camera(config, args):
    """第一步：只开相机，验证出帧、dtype、位深与行触发是否正常。"""
    processing = config.resolved_processing()
    camera = octsdk.open_camera(config.camera, pace=False)
    stats = {"packages": 0, "rows": 0, "min": None, "max": None, "saturated": 0}

    def on_package(package):
        stats["packages"] += 1
        stats["rows"] += package.rows
        data = package.spectra
        if stats["min"] is None:
            stats["min"] = float(data.min())
            stats["max"] = float(data.max())
            full_scale = (1 << int(config.camera.bit_depth)) - 1
            stats["saturated"] = int(np.count_nonzero(data >= full_scale))
        if stats["packages"] % 5 == 0:
            print(f"  ... 已收 {stats['packages']} 包 / {stats['rows']} 行")

    try:
        camera.open()
        configured = camera.configure(
            rows_per_bline=processing.x_pixels,
            spectral_samples=processing.spectral_samples,
        )
        keys = (
            "pixel_format",
            "taps",
            "image_width",
            "image_height",
            "offset_x",
            "line_rate_hz",
            "exposure_us",
            "trigger_from_ccf",
        )
        print("  生效参数:", {key: configured[key] for key in keys if key in configured})
        camera.start(
            on_package,
            x_pixels=processing.x_pixels,
            bline_avg=processing.bline_avg,
            package_y_lines=max(1, min(int(config.engine.package_y_lines), args.y_lines)),
            y_lines=args.y_lines,
        )
        deadline = time.time() + float(args.timeout)
        while camera.is_running and time.time() < deadline:
            if stats["packages"] >= int(args.packages):
                break
            time.sleep(0.02)
        camera.stop()
        info = camera.describe()
        print(
            f"  统计: frames={info['frames_delivered']} packages={stats['packages']} "
            f"rows={stats['rows']} zero_frames={info['zero_frames']} error={camera.error}"
        )
        if stats["min"] is not None:
            print(
                f"  首包数值范围: [{stats['min']:.0f}, {stats['max']:.0f}] "
                f"(满量程 {(1 << int(config.camera.bit_depth)) - 1}，饱和像素 {stats['saturated']})"
            )
        ok = (
            stats["packages"] >= 1
            and int(info["zero_frames"]) == 0
            and camera.error is None
            and stats["min"] is not None
            and stats["max"] > stats["min"]
        )
        detail = (
            f"{stats['packages']} 包 / {stats['rows']} 行"
            if ok
            else "检查：DO 是否出触发、.vlcf 行触发源、相机是否被 IKapViewer 占用"
        )
        return report("camera acquisition", ok, detail)
    except Exception as error:
        return report("camera acquisition", False, repr(error))
    finally:
        camera.close()


def step_aodo(config, args):
    """第二步：只开 DAQ，验证波形时长/幅值/通道顺序与 PFI3→PFI7 触发。"""
    processing = config.resolved_processing()
    aodo = octsdk.open_aodo(config.aodo, camera_config=config.camera)
    try:
        aodo.open()
        info = aodo.configure(config.scan, processing, octsdk.AcqModes.FINITE_CSCAN)
        spec = aodo.waveform
        print(
            f"  通道: spec={info.get('ao_channel_spec', 'n/a')} "
            f"order={info.get('channel_order', 'n/a')}"
        )
        print(
            f"  波形: samples={spec.samples} line_samples={spec.line_samples} "
            f"rate={spec.sample_rate_hz:g} Hz duration={spec.duration_s:.3f}s "
            f"do_pulses={int(np.count_nonzero(spec.do))}"
        )
        print(
            f"  幅值: X={spec.ao[0].min():+.4f}..{spec.ao[0].max():+.4f} V "
            f"Y={spec.ao[1].min():+.4f}..{spec.ao[1].max():+.4f} V "
            f"(限幅 ±{config.aodo.ao_voltage_limit_v:g} V)"
        )
        aodo.start()
        hold = (
            float(args.hold)
            if args.hold is not None
            else min(5.0, max(0.5, spec.duration_s))
        )
        print(
            f"  保持输出 {hold:.2f}s —— 示波器应看到 PFI3 起始触发、AO 长度≈"
            f"{spec.duration_s:.3f}s、PFI7 有对应边沿"
        )
        time.sleep(hold)
        aodo.stop()
        ok = spec.samples > 0 and int(np.count_nonzero(spec.do)) > 0
        return report(
            "aodo output", ok, f"channel={info.get('ao_channel_spec', 'n/a')}"
        )
    except Exception as error:
        return report("aodo output", False, repr(error))
    finally:
        aodo.close()


def step_joint(config, args):
    """第三步：引擎全链路（B-scan + C-scan），检查行数、体积形状与落盘。"""
    camera_options = {"pace": False} if config.camera.simulate else None
    engine = octsdk.OctEngine(
        config, backend=args.backend, camera_options=camera_options
    )
    try:
        with engine:
            started = time.perf_counter()
            bscan = engine.single_scan(
                octsdk.AcqModes.FINITE_BSCAN, timeout_s=args.timeout
            )
            depth_start = int(config.resolved_processing().depth_start)
            peak = (
                int(np.argmax(bscan.bline)) + depth_start
                if bscan.bline is not None
                else -1
            )
            print(
                f"  B-scan: rows={bscan.rows} packages={bscan.packages} "
                f"bline={None if bscan.bline is None else bscan.bline.shape} "
                f"peak_depth={peak} wall={bscan.timings.get('wall_s', 0.0) * 1e3:.1f} ms"
            )
            cscan = engine.single_scan(
                octsdk.AcqModes.FINITE_CSCAN,
                save_path=args.save,
                timeout_s=args.timeout,
            )
            info = cscan.describe()
            print(
                f"  C-scan: volume={info['volume_shape']} aip={info['aip_shape']} "
                f"filled_y={info['filled_y']} y_notch={info['y_notch_applied']} "
                f"saved={info['saved_path']}"
            )
        ok = (
            bscan.completed
            and cscan.completed
            and int(cscan.filled_y) == int(config.scan.y_lines)
        )
        return report(
            "joint scan", ok, f"总耗时 {time.perf_counter() - started:.2f}s"
        )
    except Exception as error:
        return report("joint scan", False, repr(error))
    finally:
        engine.close()


def main(argv=None):
    try:  # Windows 下输出被重定向/管道时按 UTF-8 写，避免中文乱码
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args(argv)
    run_all = args.all or not (args.camera_only or args.aodo_only or args.joint)
    config = build_config(args)
    print(f"=== OCT SDK bring-up（{'SIM' if args.simulate else '真机'}）===")
    step_environment(config)
    results = []
    if run_all or args.camera_only:
        print("\n[1/3] 相机采集")
        results.append(step_camera(config, args))
    if run_all or args.aodo_only:
        print("\n[2/3] 振镜 / DO 输出")
        results.append(step_aodo(config, args))
    if run_all or args.joint:
        print("\n[3/3] 联合扫描")
        results.append(step_joint(config, args))
    print(f"\n=== 结果: {sum(results)}/{len(results)} 通过 ===")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
