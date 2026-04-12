from __future__ import annotations
import json
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pydicom
from PIL import Image, ImageTk


@dataclass
class SegmentationShape:
    label: str
    shape_type: str
    points: List[Tuple[float, float]]
    area_mm2: Optional[float]
    line_color: Optional[str] = None
    fill_color: Optional[str] = None


@dataclass
class SegmentationPair:
    name: str
    dicom_path: Path
    json_path: Path


def _shoelace_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    x0, y0 = points[-1]
    for x1, y1 in points:
        area += (x0 * y1) - (x1 * y0)
        x0, y0 = x1, y1
    return abs(area) * 0.5


def _color_to_hex(color: Optional[Sequence[float]] | Optional[str], default: Optional[str]) -> Optional[str]:
    if color is None:
        return default
    if isinstance(color, str):
        return color if color.startswith('#') else default
    if isinstance(color, Sequence) and len(color) >= 3:
        try:
            r, g, b = (max(0, min(255, int(round(value)))) for value in color[:3])
        except (TypeError, ValueError):
            return default
        return f'#{r:02x}{g:02x}{b:02x}'
    return default


def _convert_spacing_from_mm(values: Sequence[float]) -> Tuple[float, float]:
    # DICOM PixelSpacing order: [row_spacing_mm, column_spacing_mm]
    row_mm, col_mm = map(float, values)
    return col_mm / 10.0, row_mm / 10.0  # x, y in cm


def _convert_physical_delta(value: Optional[float], unit_code: Optional[int]) -> Optional[float]:
    if value is None or unit_code is None:
        return None
    if unit_code == 2:  # mm
        return float(value) / 10.0
    if unit_code == 3:  # cm
        return float(value)
    return None


def extract_pixel_spacing(ds: pydicom.Dataset) -> Tuple[float, float]:
    pixel_spacing = ds.get('PixelSpacing') or ds.get('ImagerPixelSpacing')
    if pixel_spacing:
        return _convert_spacing_from_mm(pixel_spacing)

    # Ultrasound fallback via SequenceOfUltrasoundRegions
    seq = ds.get((0x0018, 0x6011))  # Sequence of Ultrasound Regions
    if seq:
        item = seq[0]
        dx = _convert_physical_delta(item.get('PhysicalDeltaX'), item.get('PhysicalUnitsXDirection'))
        dy = _convert_physical_delta(item.get('PhysicalDeltaY'), item.get('PhysicalUnitsYDirection'))
        if dx and dy:
            return dx, dy

    raise ValueError('Не удалось определить размер пикселя в DICOM-файле.')


def load_shapes(json_path: Path, spacing_x_cm: float, spacing_y_cm: float) -> List[SegmentationShape]:
    data = json.loads(json_path.read_text(encoding='utf-8'))
    spacing_x_mm = spacing_x_cm * 10.0
    spacing_y_mm = spacing_y_cm * 10.0
    default_line_color = _color_to_hex(data.get('lineColor'), '#ff6b6b')
    label_colors_raw: Dict[str, Sequence[float] | str] = data.get('label_colors', {})
    label_colors: Dict[str, Optional[str]] = {}
    for key, value in label_colors_raw.items():
        label_colors[key] = _color_to_hex(value, None)
    shapes: List[SegmentationShape] = []
    for shape in data.get('shapes', []):
        raw_points = [(float(x), float(y)) for x, y in shape.get('points', [])]
        if not raw_points:
            continue
        shape_type = (shape.get('shape_type') or 'polygon').lower()
        label = shape.get('label', 'unknown')
        label_color = label_colors.get(label)
        line_color = _color_to_hex(shape.get('line_color') or shape.get('lineColor') or label_color, default_line_color)
        fill_color_value = shape.get('fill_color') or shape.get('fillColor')
        fill_color = _color_to_hex(fill_color_value, None) if fill_color_value else None

        area_mm2: Optional[float] = None
        if shape_type == 'polygon' and len(raw_points) >= 3:
            area_mm2 = _shoelace_area(raw_points) * spacing_x_mm * spacing_y_mm
        elif shape_type == 'rectangle' and len(raw_points) >= 2:
            (x1, y1), (x2, y2) = raw_points[:2]
            area_px = abs((x2 - x1) * (y2 - y1))
            area_mm2 = area_px * spacing_x_mm * spacing_y_mm
        elif shape_type in {'circle', 'ellipse'} and len(raw_points) >= 2:
            (cx, cy), (px, py) = raw_points[:2]
            radius_x_mm = abs(px - cx) * spacing_x_mm
            radius_y_mm = abs(py - cy) * spacing_y_mm
            area_mm2 = math.pi * radius_x_mm * radius_y_mm

        shapes.append(
            SegmentationShape(
                label=label,
                shape_type=shape_type,
                points=raw_points,
                area_mm2=area_mm2,
                line_color=line_color,
                fill_color=fill_color,
            )
        )
    return shapes


def dicom_to_image(ds: pydicom.Dataset) -> Image.Image:
    array = ds.pixel_array
    if array.ndim == 2:
        mode = 'L'
    elif array.ndim == 3 and array.shape[-1] == 3:
        mode = 'RGB'
    elif array.ndim == 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
        mode = 'RGB'
    else:
        raise ValueError('Неизвестный формат пиксельных данных DICOM.')
    image = Image.fromarray(array.astype(np.uint8), mode=mode)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


class SegmentationGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('DICOM Segmentation Tool')

        self.pairs: List[SegmentationPair] = []
        self.current_pair: Optional[SegmentationPair] = None
        self.current_shapes: List[SegmentationShape] = []
        self.spacing_x_cm: Optional[float] = None
        self.spacing_y_cm: Optional[float] = None

        self.image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        self.shape_item_ids: Dict[int, int] = {}
        self.highlight_id: Optional[int] = None
        self.display_scale: float = 1.0
        self.max_canvas_width = 900
        self.max_canvas_height = 700

        self.measure_mode = False
        self.measure_points_canvas: List[Tuple[float, float]] = []
        self.measure_points_original: List[Tuple[float, float]] = []
        self.measure_marker_ids: List[int] = []
        self.measure_line_id: Optional[int] = None

        self._build_layout()

    # -------------------------- UI Layout ---------------------------------
    def _build_layout(self) -> None:
        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True)

        left = ttk.Frame(main, padding=10)
        left.pack(side='left', fill='y')

        ttk.Button(left, text='Выбрать папку', command=self.select_folder).pack(fill='x')
        self.folder_label = ttk.Label(left, text='Папка не выбрана', wraplength=200)
        self.folder_label.pack(fill='x', pady=(6, 12))

        ttk.Label(left, text='Найденные пары').pack(anchor='w')
        self.pair_list = tk.Listbox(left, width=30, height=20)
        self.pair_list.pack(fill='both', expand=True)
        self.pair_list.bind('<<ListboxSelect>>', self.on_pair_select)

        right = ttk.Frame(main, padding=10)
        right.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(right, background='black', width=800, height=600)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Button-1>', self.on_canvas_click)

        info = ttk.Frame(right)
        info.pack(fill='x', pady=(8, 0))

        ttk.Button(info, text='Линейка', command=self.start_measurement).pack(side='left')
        ttk.Button(info, text='Сброс измерения', command=self.reset_measurement).pack(side='left', padx=6)

        self.measure_label = ttk.Label(info, text='Длина: —')
        self.measure_label.pack(side='left', padx=12)

        shapes_frame = ttk.LabelFrame(right, text='Сегментации')
        shapes_frame.pack(fill='both', expand=True, pady=(8, 0))

        columns = ('label', 'area')
        self.shape_tree = ttk.Treeview(shapes_frame, columns=columns, show='headings', height=8)
        self.shape_tree.heading('label', text='Метка')
        self.shape_tree.heading('area', text='Площадь, мм²')
        self.shape_tree.column('label', width=200, anchor='w')
        self.shape_tree.column('area', width=120, anchor='center')
        self.shape_tree.pack(fill='both', expand=True)
        self.shape_tree.bind('<<TreeviewSelect>>', self.on_shape_select)

        self.status_var = tk.StringVar(value='Готово')
        ttk.Label(self.root, textvariable=self.status_var, relief='sunken', anchor='w').pack(fill='x', side='bottom')

    # -------------------------- Folder & Pair Handling ---------------------
    def select_folder(self) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return
        path = Path(folder)
        self.folder_label.configure(text=str(path))
        self.scan_pairs(path)

    def scan_pairs(self, folder: Path) -> None:
        dicoms = {p.stem: p for p in folder.glob('*.dcm')}
        jsons = {p.stem: p for p in folder.glob('*.json')}
        names = sorted(set(dicoms) & set(jsons))
        self.pairs = [SegmentationPair(name=n, dicom_path=dicoms[n], json_path=jsons[n]) for n in names]
        self.pair_list.delete(0, tk.END)
        for pair in self.pairs:
            self.pair_list.insert(tk.END, pair.name)
        if not self.pairs:
            messagebox.showwarning('Не найдено', 'В папке нет пар DICOM+JSON.')
        self.status_var.set(f'Найдено пар: {len(self.pairs)}')
        self.clear_view()

    def on_pair_select(self, event: tk.Event) -> None:
        selection = self.pair_list.curselection()
        if not selection:
            return
        pair = self.pairs[selection[0]]
        try:
            self.load_pair(pair)
        except Exception as exc:  # noqa: BLE001 - показываем пользователю
            messagebox.showerror('Ошибка', str(exc))
            self.status_var.set(f'Ошибка: {exc}')

    # -------------------------- Loading & Rendering ------------------------
    def load_pair(self, pair: SegmentationPair) -> None:
        ds = pydicom.dcmread(pair.dicom_path)
        spacing_x, spacing_y = extract_pixel_spacing(ds)
        shapes = load_shapes(pair.json_path, spacing_x, spacing_y)
        if not shapes:
            raise ValueError('В JSON нет полигонов для отображения.')

        image = dicom_to_image(ds)

        self.current_pair = pair
        self.current_shapes = shapes
        self.spacing_x_cm = spacing_x
        self.spacing_y_cm = spacing_y
        self.image = image
        self.tk_image = ImageTk.PhotoImage(image)

        self.redraw_canvas()
        self.populate_shapes()
        self.status_var.set(
            f'{pair.name}: spacing=({spacing_x*10:.4f} x {spacing_y*10:.4f}) мм, фигур {len(shapes)}'
        )

    def redraw_canvas(self) -> None:
        if not self.image:
            return
        scale = min(
            self.max_canvas_width / self.image.width,
            self.max_canvas_height / self.image.height,
            1.0,
        )
        self.display_scale = scale
        if scale != 1.0:
            display_image = self.image.resize(
                (int(self.image.width * scale), int(self.image.height * scale)), Image.Resampling.LANCZOS
            )
        else:
            display_image = self.image
        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.delete('all')
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        self.shape_item_ids.clear()
        for idx, shape in enumerate(self.current_shapes):
            color = shape.line_color or '#ff6b6b'
            fill = shape.fill_color
            coords = [(x * self.display_scale, y * self.display_scale) for x, y in shape.points]
            item_id: Optional[int] = None
            if shape.shape_type == 'polygon':
                item_id = self.canvas.create_polygon(
                    *sum(coords, ()),
                    outline=color,
                    fill=fill if fill else '',
                    width=2,
                )
            elif shape.shape_type == 'rectangle' and len(coords) >= 2:
                (x1, y1), (x2, y2) = coords[:2]
                item_id = self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    outline=color,
                    fill=fill if fill else '',
                    width=2,
                )
            elif shape.shape_type in {'circle', 'ellipse'} and len(coords) >= 2:
                (cx, cy), (px, py) = coords[:2]
                rx, ry = abs(px - cx), abs(py - cy)
                item_id = self.canvas.create_oval(
                    cx - rx,
                    cy - ry,
                    cx + rx,
                    cy + ry,
                    outline=color,
                    fill=fill if fill else '',
                    width=2,
                )
            elif shape.shape_type in {'line', 'linestrip', 'polyline'} and len(coords) >= 2:
                item_id = self.canvas.create_line(*sum(coords, ()), fill=color, width=2)
            elif shape.shape_type == 'point':
                (px, py) = coords[0]
                item_id = self.canvas.create_oval(px - 3, py - 3, px + 3, py + 3, outline=color, fill=color)
            else:
                if len(coords) >= 2:
                    item_id = self.canvas.create_line(*sum(coords, ()), fill=color, width=2)
                else:
                    (px, py) = coords[0]
                    item_id = self.canvas.create_oval(px - 3, py - 3, px + 3, py + 3, outline=color, fill=color)
            if item_id is not None:
                self.shape_item_ids[idx] = item_id
        self.highlight_id = None
        self.reset_measurement()

    def populate_shapes(self) -> None:
        self.shape_tree.delete(*self.shape_tree.get_children())
        for idx, shape in enumerate(self.current_shapes):
            area_text = f'{shape.area_mm2:.2f}' if shape.area_mm2 is not None else '—'
            self.shape_tree.insert('', 'end', iid=str(idx), values=(shape.label, area_text))

    def clear_view(self) -> None:
        self.current_pair = None
        self.current_shapes = []
        self.spacing_x_cm = None
        self.spacing_y_cm = None
        self.image = None
        self.tk_image = None
        self.display_scale = 1.0
        self.canvas.delete('all')
        self.shape_tree.delete(*self.shape_tree.get_children())
        self.measure_label.configure(text='Длина: —')

    # -------------------------- Shape selection ---------------------------
    def on_shape_select(self, event: tk.Event) -> None:
        selection = self.shape_tree.selection()
        if not selection:
            self._remove_highlight()
            return
        idx = int(selection[0])
        item_id = self.shape_item_ids.get(idx)
        if item_id is None:
            return
        self._remove_highlight()
        self.canvas.itemconfigure(item_id, width=4)
        self.highlight_id = item_id

    def _remove_highlight(self) -> None:
        if self.highlight_id is None:
            return
        self.canvas.itemconfigure(self.highlight_id, width=2)
        self.highlight_id = None

    # -------------------------- Measurement --------------------------------
    def start_measurement(self) -> None:
        if not self.spacing_x_cm or not self.spacing_y_cm:
            messagebox.showwarning('Нет данных', 'Сначала загрузите пару DICOM+JSON.')
            return
        self.measure_mode = True
        self.measure_points_canvas.clear()
        self.measure_points_original.clear()
        self.clear_measurement_graphics()
        self.status_var.set('Линейка: выберите две точки на изображении.')

    def reset_measurement(self) -> None:
        self.measure_mode = False
        self.measure_points_canvas.clear()
        self.measure_points_original.clear()
        self.clear_measurement_graphics()
        self.measure_label.configure(text='Длина: —')

    def clear_measurement_graphics(self) -> None:
        for item in self.measure_marker_ids:
            self.canvas.delete(item)
        if self.measure_line_id:
            self.canvas.delete(self.measure_line_id)
        self.measure_marker_ids.clear()
        self.measure_line_id = None

    def on_canvas_click(self, event: tk.Event) -> None:
        if not self.measure_mode:
            return
        self.measure_points_canvas.append((event.x, event.y))
        scale = self.display_scale if self.display_scale else 1.0
        self.measure_points_original.append((event.x / scale, event.y / scale))
        marker = self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='yellow')
        self.measure_marker_ids.append(marker)
        if len(self.measure_points_canvas) == 2:
            self.measure_line_id = self.canvas.create_line(
                *self.measure_points_canvas[0], *self.measure_points_canvas[1], fill='yellow', width=2
            )
            self.finish_measurement()

    def finish_measurement(self) -> None:
        if len(self.measure_points_original) != 2 or not self.spacing_x_cm or not self.spacing_y_cm:
            return
        (x0, y0), (x1, y1) = self.measure_points_original
        dx = (x1 - x0) * self.spacing_x_cm
        dy = (y1 - y0) * self.spacing_y_cm
        distance = math.hypot(dx, dy)
        self.measure_label.configure(text=f'Длина: {distance:.2f} см')
        self.status_var.set('Линейка: измерение завершено. Нажмите "Сброс" для нового измерения.')
        self.measure_mode = False

    # -------------------------- Run ---------------------------------------
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = SegmentationGUI()
    gui.run()


if __name__ == '__main__':
    main()
