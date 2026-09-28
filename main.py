"""入口：CLI 拼接 / 啟動 Tkinter GUI.

用法：
    conda activate opencv_evk
    python main.py                        # 啟動 GUI（預設 Pic/ 資料夾）
    python main.py --folder ./Pic         # 指定資料夾啟動 GUI
    python main.py --cli --save pano.jpg  # 純指令行拼接並存檔
    python main.py --cli --no-balance --save pano_nobal.jpg
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stitcher import stitch_folder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Image Stitching 專案入口")
    p.add_argument("--folder", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pic"),
                   help="影像資料夾 (預設: ./Pic)")
    p.add_argument("--width", type=int, default=1240, help="工作寬度 (預設 1240)")
    p.add_argument("--no-balance", action="store_true", help="關閉色彩平衡")
    p.add_argument("--cli", action="store_true", help="純指令行模式 (不開 GUI)")
    p.add_argument("--save", default="panorama.jpg", help="CLI 模式輸出檔名")
    p.add_argument("--method", default="auto", choices=["auto", "manual", "stitcher"],
                   help="拼接方法 (預設 auto)")
    p.add_argument("--feature", default="sift", choices=["sift", "orb"],
                   help="手刻鏈特徵器 (預設 sift；orb=Oriented FAST and Rotated BRIEF，較快)")
    p.add_argument("--seam", default="voronoi", choices=["voronoi", "none"],
                   help="手刻鏈縫線 (預設 voronoi 中線縫；none=關閉，全區混合)")
    p.add_argument("--gain-blocks", action="store_true",
                   help="分塊增益 (抗圖內亮度梯度，如天空與地面趨勢不同)")
    p.add_argument("--gui", action="store_true", help="強制 GUI 模式")
    return p.parse_args()


def run_cli(folder: str, width: int, balance: bool, save: str, method: str, feature: str, seam: str,
            gain_blocks: bool):
    res = stitch_folder(folder, work_width=width, color_balance=balance,
                        method=method, feature=feature, seam=seam,
                        gain_blocks=gain_blocks)
    cv2.imwrite(save, res.panorama)
    print(f"尺寸: {res.panorama.shape[1]}x{res.panorama.shape[0]}")
    print(f"方法: {res.method} / 特徵: {res.feature} / 縫線: {res.seam}")
    for i, n in enumerate(res.filenames):
        print(f"[{i}] {n} scale={res.scales[i]:.4f} yaw={res.yaws[i]:+.2f}deg "
              f"roll={res.rolls[i]:+.2f}deg pitch~{res.pitches[i]:+.2f}deg gain={res.gains[i]:.4f}")
        print(f"    polygon={res.polygons[i].reshape(-1).round(1).tolist()}")
    if res.inliers:
        print(f"inliers: {res.inliers}")
    print(f"已儲存: {save}")


def main():
    args = parse_args()
    balance = not args.no_balance
    if args.cli:
        run_cli(args.folder, args.width, balance, args.save, args.method, args.feature, args.seam,
                args.gain_blocks)
    else:
        from gui import StitchApp
        app = StitchApp(init_folder=args.folder)
        if args.width != 1240:
            app.width_var.set(args.width)
        if not balance:
            app.balance_var.set(False)
        if args.feature != "sift":
            app.feature_var.set(args.feature.upper())
        app.mainloop()


if __name__ == "__main__":
    main()
