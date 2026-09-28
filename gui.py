"""Tkinter 互動介面：階段五.

版面：
  左：主畫布 (Canvas) — 顯示拼接大圖 + 各圖彩色多邊形外框 (create_polygon)
  右：控制面板 — 選擇資料夾 / 工作寬度 / 色彩平衡開關 / 開始拼接 / 儲存結果 / 影像清單
  右下：雷達圖 (Radar View) — 圓心=拍攝者，扇形=各照片拍攝方向
互動：
  滾輪：以滑鼠為中心無級縮放 | 左鍵拖拽：平移 (外框連動) | 左鍵點擊：高亮邊框+狀態列顯示檔名/縮放/Yaw
"""
from __future__ import annotations

import math
import os
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from stitcher import StitchResult, stitch_folder

HFOV = 50.0  # 雷達扇形半寬所用的水平視角（度），Canvas 與存檔共用

PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
           "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe",
           "#008080", "#e6beff", "#aa6e28", "#800000", "#aaffc3"]


def _cjk_font(size: int):
    """找系統中文字型，找不到回傳 None（改用英文標註，避免 tofu 方框）。"""
    import os as _os
    for p in ("/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if _os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return None


def radar_to_image(yaws: list[float], names: list[str], colors: list[str],
                   selected: int = -1, size: int = 800) -> Image.Image:
    """把雷達逆推結果畫成 PIL 影像（俯視圖存檔用），與 RadarView 同一映射：
    yaw=0 朝上（北/基準圖方向），yaw>0 偏右（東）。
    扇形以半透明疊加（相鄰照片視角本就重疊），再疊每張的中央方向線＋編號。"""
    base = Image.new("RGBA", (size, size), "#111418ff")
    d = ImageDraw.Draw(base)
    cx = cy = size / 2
    R = size / 2 - 60
    for r in (R, R * 0.66, R * 0.33):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#2c3540ff", width=2)
    for deg in range(0, 360, 30):
        a = math.radians(deg)
        d.line([cx, cy, cx + R * math.sin(a), cy - R * math.cos(a)], fill="#1d232bff")

    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    bbox = [cx - R, cy - R, cx + R, cy + R]
    for i, yaw in enumerate(yaws):
        start = 270 + yaw - HFOV / 2  # PIL: 0°=東、順時針；北=270°
        end = 270 + yaw + HFOV / 2
        col = colors[i % len(colors)]
        od.pieslice(bbox, start=start, end=end, fill=col + "99",  # ~60% 透明
                    outline="#ffffff" if i == selected else None,
                    width=5 if i == selected else 1)
    img = Image.alpha_composite(base, overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    font_idx = _cjk_font(30) or ImageFont.load_default()
    font_cap = _cjk_font(24) or ImageFont.load_default()
    # 中央方向線＋編號（畫在最上層，不被扇形蓋掉）
    for i, yaw in enumerate(yaws):
        a = math.radians(yaw)
        ex, ey = cx + R * math.sin(a), cy - R * math.cos(a)
        d.line([cx, cy, ex, ey], fill=colors[i % len(colors)], width=3)
        tx, ty = cx + (R + 18) * math.sin(a), cy - (R + 18) * math.cos(a)
        txt = str(i)
        bb = d.textbbox((0, 0), txt, font=font_idx)
        d.text((tx - (bb[2] - bb[0]) / 2, ty - (bb[3] - bb[1]) / 2),
               txt, fill="white", font=font_idx)
    d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill="white", outline="black")
    d.text((cx - 70, 10), "N (ref=0°)", fill="#8b95a1", font=font_cap)
    cap = "圓心=拍攝者，扇形=各照片拍攝方向 (Yaw 近似值)" if _cjk_font(24) else \
        "center=photographer, sectors=shooting dirs (approx yaw)"
    d.text((12, size - 36), cap, fill="#8b95a1", font=font_cap)
    return img


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


class RadarView(tk.Canvas):
    """鳥瞰雷達圖：圓心=拍攝者，扇形=拍攝方向 (Yaw 近似值)."""

    def __init__(self, master, size: int = 220):
        super().__init__(master, width=size, height=size, bg="#111418",
                         highlightthickness=1, highlightbackground="#444")
        self.size = size

    def draw(self, yaws: list[float], names: list[str], colors: list[str],
             selected: int = -1, hfov: float = HFOV):
        self.delete("all")
        cx = cy = self.size / 2
        R = self.size / 2 - 14
        # 同心圓 + 方位刻度
        for r in (R, R * 0.66, R * 0.33):
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#2c3540")
        for deg in range(0, 360, 30):
            a = math.radians(deg)
            self.create_line(cx, cy, cx + R * math.sin(a), cy - R * math.cos(a),
                             fill="#1d232b")
        # 北 (Yaw=0) 標示
        self.create_text(cx, 8, text="N (ref=0°)", fill="#8b95a1", font=("Arial", 8))
        # 扇形：yaw 0=上方，右正左負
        for i, yaw in enumerate(yaws):
            start = 90 + (-yaw - hfov / 2) * 1.0  # canvas 角度：0=東，逆時針；轉換自上北基準
            # 簡化：用向量端點畫扇形多邊形
            a1 = math.radians(yaw - hfov / 2)
            a2 = math.radians(yaw + hfov / 2)
            pts = [cx, cy,
                   cx + R * math.sin(a1), cy - R * math.cos(a1),
                   cx + R * math.sin(a2), cy - R * math.cos(a2)]
            outline = "white" if i == selected else ""
            width = 2 if i == selected else 1
            self.create_polygon(pts, fill=colors[i % len(colors)],
                                stipple="gray50" if i != selected else "",
                                outline=outline, width=width,
                                tags=f"radar_{i}")
            mid = math.radians(yaw)
            self.create_text(cx + (R + 2) * math.sin(mid) * 0.92,
                             cy - (R + 2) * math.cos(mid) * 0.92,
                             text=str(i), fill="white", font=("Arial", 8, "bold"))
        # 拍攝者圓心
        self.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="white", outline="black")


class StitchApp(tk.Tk):
    def __init__(self, init_folder: str = ""):
        super().__init__()
        self.title("Image Stitching — 拼接 + 色彩平衡 + 拍照區域逆推")
        self.geometry("1240x760")

        self.result: StitchResult | None = None
        self.photo_bgr: np.ndarray | None = None   # 顯示用 (RGB PIL 轉回)
        self.photo_tk: ImageTk.PhotoImage | None = None
        self.view_scale = 1.0
        self.off_x = 0.0
        self.off_y = 0.0
        self.img_id = None
        self.poly_ids: list[int] = []
        self.selected = -1
        self.colors = PALETTE
        self._drag = None

        # ---- 上方工具列（兩列式，視窗窄也不會擠掉按鈕） ----
        # ---- 工具列第 1 列：資料夾 ----
        bar1 = ttk.Frame(self)
        bar1.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 0))
        ttk.Label(bar1, text="資料夾:").pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value=init_folder)
        ttk.Entry(bar1, textvariable=self.folder_var, width=52).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar1, text="選擇資料夾", command=self.select_folder).pack(side=tk.LEFT)

        # ---- 工具列第 2 列：參數 + 動作（獨立一列，視窗窄也不會被擠掉）----
        bar2 = ttk.Frame(self)
        bar2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 4))
        ttk.Label(bar2, text="工作寬度(px):").pack(side=tk.LEFT)
        self.width_var = tk.IntVar(value=1240)
        ttk.Spinbox(bar2, from_=400, to=2000, increment=100,
                    textvariable=self.width_var, width=7).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(bar2, text="降採樣拼接，越大越細但越慢",
                  foreground="gray").pack(side=tk.LEFT, padx=(4, 8))
        self.balance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar2, text="啟用色彩平衡 (Gain+Multiband)",
                        variable=self.balance_var).pack(side=tk.LEFT)
        ttk.Label(bar2, text="方法:").pack(side=tk.LEFT, padx=(8, 0))
        self.method_var = tk.StringVar(value="自動(手刻優先)")
        self.method_box = ttk.Combobox(bar2, textvariable=self.method_var, width=14,
                                       state="readonly",
                                       values=["自動(手刻優先)", "手刻鏈", "Stitcher"])
        self.method_box.pack(side=tk.LEFT, padx=2)
        ttk.Label(bar2, text="特徵:").pack(side=tk.LEFT, padx=(8, 0))
        self.feature_var = tk.StringVar(value="SIFT")
        self.feature_box = ttk.Combobox(bar2, textvariable=self.feature_var, width=6,
                                        state="readonly", values=["SIFT", "ORB"])
        self.feature_box.pack(side=tk.LEFT, padx=2)
        ttk.Button(bar2, text="開始拼接", command=self.run_stitch_async).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Button(bar2, text="儲存全景", command=self.save_pano).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar2, text="儲存俯視圖", command=self.save_radar).pack(side=tk.LEFT, padx=2)

        # ---- 主區 ----
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        self.canvas = tk.Canvas(main, bg="#22262b", highlightthickness=1,
                                highlightbackground="#555")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_wheel)       # macOS/Win
        self.canvas.bind("<Button-4>", lambda e: self.zoom(e, 1.1))   # Linux 上滾
        self.canvas.bind("<Button-5>", lambda e: self.zoom(e, 1 / 1.1))
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        side = ttk.Frame(main, width=300)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        side.pack_propagate(False)

        ttk.Label(side, text="影像清單 (點選=高亮)").pack(anchor=tk.W)
        self.listbox = tk.Listbox(side, height=9)
        self.listbox.pack(fill=tk.X)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        ttk.Label(side, text="雷達圖 (拍照區域逆推)").pack(anchor=tk.W, pady=(8, 2))
        self.radar = RadarView(side, size=220)
        self.radar.pack()
        ttk.Button(side, text="儲存俯視圖", command=self.save_radar).pack(pady=(2, 0))

        ttk.Label(side, text="匹配 inliers / 方法").pack(anchor=tk.W, pady=(8, 2))
        self.info = tk.Text(side, height=8, width=38, state=tk.DISABLED)
        self.info.pack(fill=tk.X)

        # ---- 狀態列 ----
        self.status = tk.StringVar(value="就緒：選擇 Pic 資料夾後按「開始拼接」")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    # -- 控制 --
    def select_folder(self):
        d = filedialog.askdirectory(initialdir=self.folder_var.get() or os.getcwd())
        if d:
            self.folder_var.set(d)

    METHOD_MAP = {"自動(手刻優先)": "auto", "手刻鏈": "manual", "Stitcher": "stitcher"}

    def run_stitch_async(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("錯誤", "請先選擇有效的影像資料夾")
            return
        self.status.set("拼接中 ... (特徵運算需數十秒)")
        self.update_idletasks()
        t = threading.Thread(target=self._stitch_job,
                             args=(folder, int(self.width_var.get()),
                                   bool(self.balance_var.get()),
                                   self.METHOD_MAP.get(self.method_var.get(), "auto"),
                                   self.feature_var.get().lower()),
                             daemon=True)
        t.start()

    def _stitch_job(self, folder, width, balance, method="auto", feature="sift"):
        try:
            res = stitch_folder(folder, work_width=width, color_balance=balance,
                                method=method, feature=feature)
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda: messagebox.showerror("拼接失敗", str(e)))
            self.after(0, lambda: self.status.set(f"失敗：{e}"))
            return
        self.after(0, lambda: self.show_result(res))

    def show_result(self, res: StitchResult):
        self.result = res
        self.selected = -1
        rgb = cv2.cvtColor(res.panorama, cv2.COLOR_BGR2RGB)
        self.photo_bgr = rgb
        self.listbox.delete(0, tk.END)
        for i, n in enumerate(res.filenames):
            self.listbox.insert(tk.END,
                                f"{i}: {n}  scale={res.scales[i]:.3f}  yaw={res.yaws[i]:+.1f}°")
        self.fit_view()
        self.redraw()
        self.radar.draw(res.yaws, res.filenames,
                        [self.colors[i % len(self.colors)] for i in range(len(res.filenames))])
        inl = "/".join(map(str, res.inliers)) if res.inliers else "-"
        self.set_info(f"方法: {res.method}\n特徵: {res.feature}\n全景: {res.panorama.shape[1]}x{res.panorama.shape[0]}\n"
                      f"inliers(相鄰段): {inl}\n"
                      f"gains: {' '.join(f'{g:.3f}' for g in res.gains)}\n"
                      f"色彩平衡: {'ON' if self.balance_var.get() else 'OFF'}")
        self.status.set(f"完成：{len(res.filenames)} 張，{res.method}，滾輪縮放 / 拖拽平移 / 點擊高亮")

    def set_info(self, text: str):
        self.info.config(state=tk.NORMAL)
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, text)
        self.info.config(state=tk.DISABLED)

    def radar_colors(self) -> list[str]:
        n = len(self.result.filenames) if self.result else 0
        return [self.colors[i % len(self.colors)] for i in range(n)]

    def save_radar(self):
        """把雷達逆推俯視圖存成 PNG/JPG（與畫面同映射、可重現）。"""
        if self.result is None:
            messagebox.showinfo("提示", "尚無拼接結果")
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not p:
            return
        img = radar_to_image(self.result.yaws, self.result.filenames,
                             self.radar_colors(), selected=self.selected, size=800)
        img.save(p)
        self.status.set(f"俯視圖已儲存：{p}")

    def save_pano(self):
        if self.result is None:
            messagebox.showinfo("提示", "尚無拼接結果")
            return
        p = filedialog.asksaveasfilename(defaultextension=".jpg",
                                         filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if p:
            cv2.imwrite(p, self.result.panorama)
            self.status.set(f"已儲存：{p}")

    # -- 視圖變換 --
    def fit_view(self):
        if self.photo_bgr is None:
            return
        self.canvas.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ph, pw = self.photo_bgr.shape[:2]
        self.view_scale = min(cw / pw, ch / ph, 1.0) or 0.2
        self.off_x = (cw - pw * self.view_scale) / 2
        self.off_y = (ch - ph * self.view_scale) / 2

    def w2c(self, x, y):
        return x * self.view_scale + self.off_x, y * self.view_scale + self.off_y

    def c2w(self, x, y):
        return (x - self.off_x) / self.view_scale, (y - self.off_y) / self.view_scale

    def redraw(self):
        self.canvas.delete("all")
        self.poly_ids = []
        if self.photo_bgr is None or self.result is None:
            self.canvas.create_text(300, 200, text="尚無結果", fill="gray")
            return
        ph, pw = self.photo_bgr.shape[:2]
        dw, dh = max(1, int(pw * self.view_scale)), max(1, int(ph * self.view_scale))
        small = Image.fromarray(self.photo_bgr).resize((dw, dh), Image.BILINEAR)
        self.photo_tk = ImageTk.PhotoImage(small)
        self.img_id = self.canvas.create_image(self.off_x, self.off_y,
                                               anchor=tk.NW, image=self.photo_tk)
        for i, poly in enumerate(self.result.polygons):
            pts = []
            for x, y in poly:
                cx, cy = self.w2c(x, y)
                pts.extend([cx, cy])
            col = self.colors[i % len(self.colors)]
            width = 3 if i == self.selected else 1
            pid = self.canvas.create_polygon(pts, outline=col, fill="",
                                             width=width, tags=f"poly_{i}")
            self.poly_ids.append(pid)
            # 影像編號標籤放在多邊形中心
            ccx, ccy = self.w2c(*np.mean(poly, axis=0))
            self.canvas.create_text(ccx, ccy, text=str(i), fill=col,
                                    font=("Arial", 10, "bold"))

    # -- 互動 --
    def on_wheel(self, e):
        self.zoom(e, 1.1 if e.delta > 0 else 1 / 1.1)

    def zoom(self, e, factor: float):
        if self.photo_bgr is None:
            return
        wx, wy = self.c2w(e.x, e.y)
        self.view_scale = min(8.0, max(0.05, self.view_scale * factor))
        self.off_x = e.x - wx * self.view_scale
        self.off_y = e.y - wy * self.view_scale
        self.redraw()

    def on_press(self, e):
        self._drag = (e.x, e.y, self.off_x, self.off_y, False)

    def on_drag(self, e):
        if not self._drag:
            return
        x0, y0, ox, oy, _ = self._drag
        if abs(e.x - x0) + abs(e.y - y0) > 3:
            self._drag = (x0, y0, ox, oy, True)
        self.off_x = ox + (e.x - x0)
        self.off_y = oy + (e.y - y0)
        self.redraw()

    def on_release(self, e):
        if not self._drag:
            return
        _, _, _, _, moved = self._drag
        self._drag = None
        if not moved:
            self.click_select(e.x, e.y)

    def click_select(self, cx, cy):
        if self.result is None:
            return
        wx, wy = self.c2w(cx, cy)
        hit = -1
        for i in range(len(self.result.polygons) - 1, -1, -1):  # 上層優先
            if point_in_poly(wx, wy, self.result.polygons[i]):
                hit = i
                break
        self.selected = hit
        self.redraw()
        if hit >= 0:
            r = self.result
            self.status.set(f"選中 [{hit}] {r.filenames[hit]} ｜ scale={r.scales[hit]:.3f} ｜ "
                            f"yaw={r.yaws[hit]:+.1f}° ｜ gain={r.gains[hit]:.3f}")
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(hit)
            self.radar.draw(r.yaws, r.filenames,
                            [self.colors[i % len(self.colors)] for i in range(len(r.filenames))],
                            selected=hit)
        else:
            self.status.set("未點中任何影像區域")

    def on_list_select(self, _e):
        sel = self.listbox.curselection()
        if not sel or self.result is None:
            return
        self.selected = int(sel[0])
        self.redraw()
        r = self.result
        i = self.selected
        self.status.set(f"選中 [{i}] {r.filenames[i]} ｜ scale={r.scales[i]:.3f} ｜ yaw={r.yaws[i]:+.1f}°")
        self.radar.draw(r.yaws, r.filenames,
                        [self.colors[k % len(self.colors)] for k in range(len(r.filenames))],
                        selected=i)
