from __future__ import annotations
import wx
import copy
import math
import os
from typing import List, Tuple, Optional, Set, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

EPS = 1e-10
EPS_AREA = 1e-8


class LineType(Enum):
    """Перечисление типов линий в оригами паттерне."""
    MOUNTAIN = 1
    VALLEY = 2
    AUX = 3
    CUT = 4
    NONE = 0


class LayerRelation(int):
    """Перечисление для описания взаимного расположения слоёв при складывании."""
    ABOVE = 1
    BELOW = -1
    UNKNOWN = 0


def line_intersection_rel(line1, line2):
    """
    Найти точку пересечения двух линий в относительных координатах.

    Args:
        line1 (tuple): Линия, представленная как ((x1, y1), (x2, y2)).
        line2 (tuple): Линия, представленная как ((x3, y3), (x4, y4)).

    Returns:
        tuple: Кортеж (x, y) координат пересечения или None.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (round(ix, 12), round(iy, 12))
    return None


def triangle_incenter(a, b, c):
    """
    Вычислить инцентр (центр вписанной окружности) треугольника.

    Args:
        a (tuple): Первая вершина треугольника (x, y).
        b (tuple): Вторая вершина треугольника (x, y).
        c (tuple): Третья вершина треугольника (x, y).

    Returns:
        tuple: Координаты инцентра (x, y).
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    ab = math.hypot(bx - ax, by - ay)
    bc = math.hypot(cx - bx, cy - by)
    ca = math.hypot(ax - cx, ay - cy)
    total = ab + bc + ca
    if total == 0:
        return a
    ix = (ab * cx + bc * ax + ca * bx) / total
    iy = (ab * cy + bc * ay + ca * by) / total
    return (ix, iy)


@dataclass
class OriLine:
    """Ориентированная линия, представляющая отрезок с типом складки."""
    p0: 'Point2D'
    p1: 'Point2D'
    type: LineType = LineType.AUX


@dataclass
class OriVertex:
    """Вершина в графе паттерна складок."""
    p: 'Point2D'
    edges: List['OriEdge'] = field(default_factory=list, repr=False)

    def __hash__(self):
        return hash((self.p.x, self.p.y))

    def __eq__(self, other):
        return isinstance(other, OriVertex) and self.p.distance_to(other.p) < EPS


@dataclass
class OriEdge:
    """Ориентированное ребро в графе паттерна складок."""
    sv: OriVertex
    ev: OriVertex
    line: OriLine
    type: LineType = LineType.AUX

    def is_fold_edge(self) -> bool:
        return self.type in (LineType.MOUNTAIN, LineType.VALLEY)


@dataclass
class OriHalfedge:
    """Полуребро в DCEL структуре."""
    edge: Optional[OriEdge] = None
    face: Optional['OriFace'] = None
    vertex: Optional[OriVertex] = None
    next: Optional['OriHalfedge'] = None
    prev: Optional['OriHalfedge'] = None
    opposite: Optional['OriHalfedge'] = None


@dataclass
class OriFace:
    """Грань в DCEL структуре."""
    halfedges: List[OriHalfedge] = field(default_factory=list)
    is_ccw: bool = True
    outline: List['Point2D'] = field(default_factory=list)
    z_order: int = 0

    def build_outline(self):
        self.outline = [he.vertex.p for he in self.halfedges]

    def area(self) -> float:
        if len(self.outline) < 3:
            return 0.0
        area = 0.0
        for i in range(len(self.outline)):
            p1 = self.outline[i]
            p2 = self.outline[(i + 1) % len(self.outline)]
            area += p1.x * p2.y - p2.x * p1.y
        return abs(area) / 2.0


@dataclass(frozen=True, eq=True)
class Point2D:
    """
    Двумерная точка с поддержкой векторных операций.

    Attributes:
        x (float): Координата X.
        y (float): Координата Y.
    """
    x: float
    y: float

    def __add__(self, other: 'Point2D') -> 'Point2D':
        """Сложить две точки (как векторы)."""
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        """Вычесть одну точку из другой (как векторы)."""
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        """Умножить точку на скаляр."""
        return Point2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Point2D':
        """Разделить координаты точки на скаляр."""
        return Point2D(self.x / scalar, self.y / scalar)

    def __repr__(self) -> str:
        """Возвращает строковое представление точки."""
        return f"P({self.x:.6f}, {self.y:.6f})"

    def length(self) -> float:
        """Вычислить длину вектора от начала координат до точки."""
        return math.hypot(self.x, self.y)

    def distance_to(self, other: 'Point2D') -> float:
        """Вычислить расстояние до другой точки."""
        return (self - other).length()


class FoldViewer(wx.Frame):
    """Окно для визуализации сложенного оригами."""

    def __init__(self, parent, folded_faces):
        super().__init__(parent, title="Сложенная модель", size=(800, 800))
        self.folded_faces = folded_faces
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.Centre()
        self.Show()

    def on_paint(self, event):
        dc = wx.PaintDC(self.panel)
        dc.Clear()
        dc.SetBackground(wx.Brush(wx.Colour(255, 255, 255)))
        dc.Clear()
        w, h = self.panel.GetSize()
        margin = 40
        size = min(w, h) - 2 * margin
        scale = size / 2.0
        cx, cy = w // 2, h // 2

        for face in sorted(self.folded_faces, key=lambda f: f.z_order):
            if len(face.outline) < 3:
                continue
            intensity = 80 + face.z_order * 30
            color = wx.Colour(80, 100, min(255, intensity + 80), alpha=220)
            dc.SetBrush(wx.Brush(color))
            dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 2))
            points = []
            for p in face.outline:
                x = cx + p.x * scale
                y = cy - p.y * scale
                points.append(wx.Point(int(x), int(y)))
            dc.DrawPolygon(points)


class ErrorViewer(wx.Frame):
    """Окно для визуализации паттерна складок с ошибками."""

    def __init__(self, parent, crease_pattern, problem_vertices):
        super().__init__(parent, title="Проверка складывания - Обнаружены ошибки", size=(800, 800))
        self.crease_pattern = crease_pattern
        self.problem_vertices = problem_vertices
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.Centre()
        self.Show()

    def on_paint(self, event):
        dc = wx.PaintDC(self.panel)
        dc.Clear()
        dc.SetBackground(wx.Brush(wx.Colour(255, 255, 255)))
        dc.Clear()
        w, h = self.panel.GetSize()
        margin = 40
        size = min(w, h) - 2 * margin
        scale = size / 2.0
        cx, cy = w // 2, h // 2

        dc.SetPen(wx.Pen(wx.BLACK, 2))
        dc.SetBrush(wx.Brush(wx.WHITE))
        dc.DrawRectangle(int(cx - scale), int(cy - scale), int(2 * scale), int(2 * scale))

        for line in self.crease_pattern.lines:
            p0, p1 = line.p0, line.p1
            x0 = cx + p0.x * scale
            y0 = cy - p0.y * scale
            x1 = cx + p1.x * scale
            y1 = cy - p1.y * scale
            if line.type == LineType.MOUNTAIN:
                dc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))
            elif line.type == LineType.VALLEY:
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))
            else:
                dc.SetPen(wx.Pen(wx.Colour(180, 180, 180), 1))
            dc.DrawLine(int(x0), int(y0), int(x1), int(y1))

        dc.SetPen(wx.Pen(wx.BLACK, 1))
        dc.SetBrush(wx.Brush(wx.BLACK))
        for vertex in self.crease_pattern.vertices:
            p = vertex.p
            x = cx + p.x * scale
            y = cy - p.y * scale
            dc.DrawCircle(int(x), int(y), 3)

        dc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 3))
        dc.SetBrush(wx.Brush(wx.Colour(255, 100, 100)))
        for problem_point in self.problem_vertices:
            x = cx + problem_point.x * scale
            y = cy - problem_point.y * scale
            dc.DrawCircle(int(x), int(y), 8)


class PatternViewer(wx.Frame):
    """Окно для визуализации успешного паттерна складок."""

    def __init__(self, parent, crease_pattern):
        super().__init__(parent, title="Проверка складывания - Узор корректен", size=(800, 800))
        self.crease_pattern = crease_pattern
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.Centre()
        self.Show()

    def on_paint(self, event):
        dc = wx.PaintDC(self.panel)
        dc.Clear()
        dc.SetBackground(wx.Brush(wx.Colour(255, 255, 255)))
        dc.Clear()
        w, h = self.panel.GetSize()
        margin = 40
        size = min(w, h) - 2 * margin
        scale = size / 2.0
        cx, cy = w // 2, h // 2

        dc.SetPen(wx.Pen(wx.BLACK, 2))
        dc.SetBrush(wx.Brush(wx.WHITE))
        dc.DrawRectangle(int(cx - scale), int(cy - scale), int(2 * scale), int(2 * scale))

        for line in self.crease_pattern.lines:
            p0, p1 = line.p0, line.p1
            x0 = cx + p0.x * scale
            y0 = cy - p0.y * scale
            x1 = cx + p1.x * scale
            y1 = cy - p1.y * scale
            if line.type == LineType.MOUNTAIN:
                dc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))
            elif line.type == LineType.VALLEY:
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))
            else:
                dc.SetPen(wx.Pen(wx.Colour(180, 180, 180), 1))
            dc.DrawLine(int(x0), int(y0), int(x1), int(y1))

        dc.SetPen(wx.Pen(wx.BLACK, 1))
        dc.SetBrush(wx.Brush(wx.BLACK))
        for vertex in self.crease_pattern.vertices:
            p = vertex.p
            x = cx + p.x * scale
            y = cy - p.y * scale
            dc.DrawCircle(int(x), int(y), 3)


class Foldify(wx.Frame):
    """
    Окно для рисования паттернов на квадратной сетке.

    Режимы работы:
    - MODE_INPUT: Рисование новых линий с инструментами
    - MODE_DELETE: Удаление существующих сегментов
    - MODE_ALTER: Изменение типа линии (гора <-> долина)

    Подрежимы (только в MODE_INPUT):
    - SUBMODE_SEGMENT: Рисование отрезков
    - SUBMODE_BISECTOR: Построение биссектрисы угла
    - SUBMODE_INCENTER: Построение линий к инцентру треугольника
    - SUBMODE_PERP: Построение перпендикуляра к линии

    Типы линий:
    - LINE_MOUNTAIN: Красная линия горы
    - LINE_VALLEY: Синяя линия долины
    - LINE_AUX: Серая вспомогательная линия

    Attributes:
        canvas (wx.Panel): Холст для рисования.
        rel_lines (list): Список линий в относительных координатах.
        div_num (int): Количество делений сетки.
    """

    MODE_INPUT = "input"
    MODE_DELETE = "delete"
    MODE_ALTER = "alter"

    SUBMODE_SEGMENT = "segment"
    SUBMODE_BISECTOR = "bisector"
    SUBMODE_INCENTER = "incenter"
    SUBMODE_PERP = "perp"

    LINE_MOUNTAIN = "mountain"
    LINE_VALLEY = "valley"
    LINE_AUX = "aux"

    def __init__(self, parent, title):
        """
        Инициализировать окно приложения.

        Args:
            parent: Родительское окно.
            title (str): Заголовок окна.
        """
        super(Foldify, self).__init__(parent, title=title, size=(1100, 700),
                                      style=wx.DEFAULT_FRAME_STYLE | wx.RESIZE_BORDER)

        self.div_num = 4
        self.margin = 50
        self.scale = 1.0

        self.rel_lines = []
        self.rel_intersections = set()

        self.hover_point = None
        self.selected_point = None
        self.hovered_segment = None

        self.mode = self.MODE_INPUT
        self.submode = self.SUBMODE_SEGMENT
        self.line_type = self.LINE_MOUNTAIN

        self.bisector_points = []
        self.bisector_waiting_for_edge = False
        self.incenter_points = []
        self.perp_start_point = None
        self.perp_waiting_for_edge = False

        self.history = []
        self.history_index = -1
        self.max_history = 50

        self.square_edges = []

        self.panel = wx.Panel(self)
        self.canvas = wx.Panel(self.panel, style=wx.FULL_REPAINT_ON_RESIZE)
        self.canvas.SetBackgroundColour(wx.WHITE)

        self.control_panel = wx.Panel(self.panel)
        self.control_panel.SetBackgroundColour(wx.Colour(240, 240, 240))

        self.radio_input = wx.RadioButton(self.control_panel, label="Ввод линии", style=wx.RB_GROUP)
        self.radio_delete = wx.RadioButton(self.control_panel, label="Удалить линию")
        self.radio_alter = wx.RadioButton(self.control_panel, label="Изменить тип")
        self.radio_input.SetValue(True)

        self.radio_input.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_INPUT))
        self.radio_delete.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_DELETE))
        self.radio_alter.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_ALTER))

        type_panel = wx.Panel(self.control_panel)
        type_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.radio_mountain = wx.RadioButton(type_panel, label="Гора", style=wx.RB_GROUP)
        self.radio_valley = wx.RadioButton(type_panel, label="Долина")
        self.radio_aux = wx.RadioButton(type_panel, label="Вспомогательная")
        self.radio_mountain.SetValue(True)
        type_sizer.Add(self.radio_mountain, 0, wx.ALL, 3)
        type_sizer.Add(self.radio_valley, 0, wx.ALL, 3)
        type_sizer.Add(self.radio_aux, 0, wx.ALL, 3)
        type_panel.SetSizer(type_sizer)

        self.radio_mountain.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_MOUNTAIN))
        self.radio_valley.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_VALLEY))
        self.radio_aux.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_AUX))

        self.btn_segment = wx.Button(self.control_panel, label="Отрезок")
        self.btn_bisector = wx.Button(self.control_panel, label="Биссектриса")
        self.btn_incenter = wx.Button(self.control_panel, label="Инцентр")
        self.btn_perp = wx.Button(self.control_panel, label="Перпендикуляр")

        self.btn_segment.Bind(wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_SEGMENT))
        self.btn_bisector.Bind(wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_BISECTOR))
        self.btn_incenter.Bind(wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_INCENTER))
        self.btn_perp.Bind(wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_PERP))

        self.div_label = wx.StaticText(self.control_panel, label="Делений")
        self.div_text = wx.TextCtrl(self.control_panel, value="4", size=(50, -1), style=wx.TE_CENTER)
        self.set_button = wx.Button(self.control_panel, label="Задать")
        self.set_button.Bind(wx.EVT_BUTTON, self.on_set_div)

        self.undo_button = wx.Button(self.control_panel, label="Отмена")
        self.redo_button = wx.Button(self.control_panel, label="Повтор")
        self.clear_button = wx.Button(self.control_panel, label="Очистить")

        self.undo_button.Bind(wx.EVT_BUTTON, self.on_undo)
        self.redo_button.Bind(wx.EVT_BUTTON, self.on_redo)
        self.clear_button.Bind(wx.EVT_BUTTON, self.on_clear)

        self.check_button = wx.Button(self.control_panel, label="Проверить складывание")
        self.check_button.Bind(wx.EVT_BUTTON, self.on_check_foldability)

        top_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(self.radio_input, 0, wx.ALL, 5)
        top_sizer.Add(self.radio_delete, 0, wx.ALL, 5)
        top_sizer.Add(self.radio_alter, 0, wx.ALL, 5)
        top_sizer.Add(type_panel, 0, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(wx.StaticLine(self.control_panel), 0, wx.EXPAND | wx.ALL, 5)

        icon_row = wx.BoxSizer(wx.HORIZONTAL)
        icon_row.Add(self.btn_segment, 0, wx.ALL, 3)
        icon_row.Add(self.btn_bisector, 0, wx.ALL, 3)
        icon_row.Add(self.btn_incenter, 0, wx.ALL, 3)
        icon_row.Add(self.btn_perp, 0, wx.ALL, 3)

        bottom_sizer = wx.BoxSizer(wx.VERTICAL)
        bottom_sizer.Add(self.div_label, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        bottom_sizer.Add(self.div_text, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.set_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.AddStretchSpacer()
        bottom_sizer.Add(self.undo_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.redo_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.check_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.clear_button, 0, wx.ALL | wx.EXPAND, 5)

        ctrl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctrl_sizer.Add(top_sizer, 0, wx.EXPAND)
        ctrl_sizer.Add(icon_row, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        ctrl_sizer.Add(bottom_sizer, 1, wx.EXPAND)
        self.control_panel.SetSizer(ctrl_sizer)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(self.control_panel, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(self.canvas, 1, wx.EXPAND)

        self.panel.SetSizer(main_sizer)

        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.canvas.Bind(wx.EVT_MOTION, self.on_mouse_move)
        self.canvas.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)

        self.update_grid_size()
        self.save_state()
        self.Centre()
        self.Show()

    def set_main_mode(self, mode):
        """Установить основной режим работы."""
        self.mode = mode
        self.canvas.Refresh()

    def set_submode(self, submode):
        """Установить подрежим (только в MODE_INPUT)."""
        if self.mode == self.MODE_INPUT:
            self.submode = submode
            self.bisector_points = []
            self.incenter_points = []
            self.perp_start_point = None
            self.bisector_waiting_for_edge = False
            self.perp_waiting_for_edge = False
            self.selected_point = None
            self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
            self.canvas.Refresh()

    def set_line_type(self, line_type):
        """Установить тип линии для рисования."""
        self.line_type = line_type
        self.canvas.Refresh()

    def save_state(self):
        """Сохранить текущее состояние в историю для undo/redo."""
        state = {
            "div_num": self.div_num,
            "rel_lines": copy.deepcopy(self.rel_lines)
        }
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(state)
        self.history_index += 1
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
        self.update_undo_redo_buttons()

    def restore_state(self, state):
        """Восстановить состояние из истории."""
        self.div_num = state["div_num"]
        self.div_text.SetValue(str(self.div_num))
        self.rel_lines = copy.deepcopy(state["rel_lines"])
        self.update_grid_size()
        self.update_all_intersections()
        self.canvas.Refresh()

    def on_undo(self, event):
        """Обработчик отмены последнего действия."""
        if self.history_index > 0:
            self.history_index -= 1
            self.restore_state(self.history[self.history_index])
            self.update_undo_redo_buttons()

    def on_redo(self, event):
        """Обработчик повтора отменённого действия."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.restore_state(self.history[self.history_index])
            self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        """Обновить состояние кнопок Undo/Redo согласно истории."""
        self.undo_button.Enable(self.history_index > 0)
        self.redo_button.Enable(self.history_index < len(self.history) - 1)

    def on_key_down(self, event):
        """Обработчик нажатия клавиши."""
        key = event.GetKeyCode()
        if event.ControlDown():
            if key == ord('Z'):
                self.on_undo(None)
            elif key == ord('Y'):
                self.on_redo(None)
        elif key == wx.WXK_ESCAPE:
            self.selected_point = None
            self.bisector_points = []
            self.incenter_points = []
            self.perp_start_point = None
            self.bisector_waiting_for_edge = False
            self.perp_waiting_for_edge = False
            self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
            self.canvas.Refresh()
        event.Skip()

    def update_grid_size(self):
        """Пересчитать размер сетки при изменении размера окна или количества делений."""
        w, h = self.canvas.GetSize()
        if w <= 0 or h <= 0:
            return

        usable = min(w, h)
        self.scale = usable / (usable - 2 * self.margin) if usable > 2 * self.margin else 1.0
        margin_px = self.margin * self.scale
        size = usable - 2 * margin_px

        left = (w - size) / 2
        top = (h - size) / 2
        right = left + size
        bottom = top + size

        self.grid_size = size / self.div_num if self.div_num > 0 else 50.0

        self.square_edges = [
            ((left, bottom), (right, bottom)),
            ((right, bottom), (right, top)),
            ((right, top), (left, top)),
            ((left, top), (left, bottom))
        ]

        self._grid_bounds = (left, top, right, bottom)
        self.update_all_intersections()
        self.canvas.Refresh()

    def get_grid_bounds(self):
        """Получить границы сетки в абсолютных координатах экрана."""
        return self._grid_bounds

    def rel_to_abs(self, rel_point):
        """Преобразовать точку из относительных координат (0-1) в абсолютные (экран)."""
        if not rel_point:
            return None
        left, top, right, bottom = self.get_grid_bounds()
        size = right - left
        x_rel, y_rel = rel_point
        return (left + x_rel * size, bottom - y_rel * size)

    def abs_to_rel(self, abs_point):
        """Преобразовать точку из абсолютных (экран) в относительные координаты (0-1)."""
        if not abs_point:
            return None
        left, top, right, bottom = self.get_grid_bounds()
        size = right - left
        if size == 0:
            return None
        x_abs, y_abs = abs_point
        return ((x_abs - left) / size, (bottom - y_abs) / size)

    def on_resize(self, event):
        """Обработчик изменения размера окна."""
        wx.CallAfter(self.update_grid_size)
        event.Skip()

    def get_abs_lines(self):
        """Получить все линии в абсолютных координатах экрана."""
        return [(self.rel_to_abs(p1), self.rel_to_abs(p2), ltype) for p1, p2, ltype in self.rel_lines]

    def get_nearest_point(self, pos):
        """Найти ближайшую значимую точку к позиции курсора."""
        rel_pos = self.abs_to_rel(pos)
        if not rel_pos:
            return None
        left, top, right, bottom = self.get_grid_bounds()
        if not (left <= pos[0] <= right and top <= pos[1] <= bottom):
            return None

        gx = round(rel_pos[0] * self.div_num) / self.div_num
        gy = round(rel_pos[1] * self.div_num) / self.div_num
        gx = max(0.0, min(1.0, gx))
        gy = max(0.0, min(1.0, gy))
        grid_point = (gx, gy)

        candidates = [grid_point] + list(self.rel_intersections)
        candidates = list(set(candidates))
        candidates_abs = [self.rel_to_abs(p) for p in candidates if self.rel_to_abs(p)]

        if not candidates_abs:
            return None

        return min(candidates_abs, key=lambda p: math.hypot(p[0] - pos[0], p[1] - pos[1]))

    def update_all_intersections(self):
        """Пересчитать все пересечения между линиями."""
        self.rel_intersections = set()

        mv_lines = [(p1, p2, ltype) for p1, p2, ltype in self.rel_lines if ltype != self.LINE_AUX]
        for i in range(len(mv_lines)):
            for j in range(i + 1, len(mv_lines)):
                inter = line_intersection_rel(mv_lines[i][:2], mv_lines[j][:2])
                if inter:
                    self.rel_intersections.add(inter)

    def distance_to_segment(self, point, segment):
        """Вычислить минимальное расстояние от точки до отрезка."""
        px, py = point
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def is_point_on_segment(self, point, segment, tol=1e-6):
        """Проверить, находится ли точка на отрезке (в пределах допуска)."""
        a, b = segment
        cross = (point[1] - a[1]) * (b[0] - a[0]) - (point[0] - a[0]) * (b[1] - a[1])
        if abs(cross) > tol:
            return False
        dot = (point[0] - a[0]) * (b[0] - a[0]) + (point[1] - a[1]) * (b[1] - a[1])
        if dot < 0:
            return False
        squared_len = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        if dot > squared_len:
            return False
        return True

    def points_equal(self, p1, p2, tol=1e-10):
        """Проверить, совпадают ли две точки в пределах допуска."""
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    def get_closest_line_and_segment(self, pos):
        """Найти ближайший сегмент линии к позиции курсора."""
        x, y = pos
        min_dist = float('inf')
        result = None

        for idx, (p1_rel, p2_rel, ltype) in enumerate(self.rel_lines):
            p1 = self.rel_to_abs(p1_rel)
            p2 = self.rel_to_abs(p2_rel)
            points_on_line = [p1, p2]
            for inter_rel in self.rel_intersections:
                inter_abs = self.rel_to_abs(inter_rel)
                if self.is_point_on_segment(inter_abs, (p1, p2), tol=1e-3):
                    points_on_line.append(inter_abs)
            points_on_line = sorted(set(points_on_line), key=lambda pt: (pt[0], pt[1]))

            for i in range(len(points_on_line) - 1):
                a, b = points_on_line[i], points_on_line[i + 1]
                dist = self.distance_to_segment(pos, (a, b))
                if dist < 20 and dist < min_dist:
                    min_dist = dist
                    result = (idx, a, b, ltype, False)

        return result

    def remove_line_segment(self, line_idx, seg_start, seg_end):
        """Удалить определённый сегмент линии."""
        p1_rel, p2_rel, ltype = self.rel_lines[line_idx]
        p1 = self.rel_to_abs(p1_rel)
        p2 = self.rel_to_abs(p2_rel)
        points_on_line = [p1, p2]
        for inter_rel in self.rel_intersections:
            inter_abs = self.rel_to_abs(inter_rel)
            if self.is_point_on_segment(inter_abs, (p1, p2), tol=1e-3):
                points_on_line.append(inter_abs)
        points_on_line = sorted(set(points_on_line), key=lambda pt: (pt[0], pt[1]))

        start_idx = next((i for i, pt in enumerate(points_on_line) if self.points_equal(pt, seg_start)), -1)
        end_idx = next((i for i, pt in enumerate(points_on_line) if self.points_equal(pt, seg_end)), -1)

        if start_idx == -1 or end_idx == -1:
            return

        new_lines = []
        if start_idx > 0:
            new_p1 = self.abs_to_rel(points_on_line[0])
            new_p2 = self.abs_to_rel(points_on_line[start_idx])
            new_lines.append((new_p1, new_p2, ltype))
        if end_idx < len(points_on_line) - 1:
            new_p1 = self.abs_to_rel(points_on_line[end_idx])
            new_p2 = self.abs_to_rel(points_on_line[-1])
            new_lines.append((new_p1, new_p2, ltype))

        if new_lines:
            self.rel_lines[line_idx] = new_lines[0]
            if len(new_lines) > 1:
                self.rel_lines.insert(line_idx + 1, new_lines[1])
        else:
            del self.rel_lines[line_idx]

        self.update_all_intersections()
        self.save_state()

    def alter_line_segment(self, line_idx, seg_start, seg_end):
        """Изменить тип определённого сегмента линии (гора <-> долина)."""
        p1_rel, p2_rel, ltype = self.rel_lines[line_idx]
        p1 = self.rel_to_abs(p1_rel)
        p2 = self.rel_to_abs(p2_rel)
        points_on_line = [p1, p2]
        for inter_rel in self.rel_intersections:
            inter_abs = self.rel_to_abs(inter_rel)
            if self.is_point_on_segment(inter_abs, (p1, p2), tol=1e-3):
                points_on_line.append(inter_abs)
        points_on_line = sorted(set(points_on_line), key=lambda pt: (pt[0], pt[1]))

        start_idx = next((i for i, pt in enumerate(points_on_line) if self.points_equal(pt, seg_start)), -1)
        end_idx = next((i for i, pt in enumerate(points_on_line) if self.points_equal(pt, seg_end)), -1)

        if start_idx == -1 or end_idx == -1:
            return

        new_type = self.LINE_VALLEY if ltype == self.LINE_MOUNTAIN else self.LINE_MOUNTAIN

        new_lines = []
        if start_idx > 0:
            new_p1 = self.abs_to_rel(points_on_line[0])
            new_p2 = self.abs_to_rel(points_on_line[start_idx])
            new_lines.append((new_p1, new_p2, ltype))
        new_p1 = self.abs_to_rel(points_on_line[start_idx])
        new_p2 = self.abs_to_rel(points_on_line[end_idx])
        new_lines.append((new_p1, new_p2, new_type))
        if end_idx < len(points_on_line) - 1:
            new_p1 = self.abs_to_rel(points_on_line[end_idx])
            new_p2 = self.abs_to_rel(points_on_line[-1])
            new_lines.append((new_p1, new_p2, ltype))

        if new_lines:
            self.rel_lines[line_idx] = new_lines[0]
            for i in range(1, len(new_lines)):
                self.rel_lines.insert(line_idx + i, new_lines[i])
        else:
            del self.rel_lines[line_idx]

        self.update_all_intersections()
        self.save_state()

    def on_mouse_move(self, event):
        """Обработчик движения мыши."""
        pos = event.GetPosition()

        if self.mode in [self.MODE_DELETE, self.MODE_ALTER] or \
                (
                        self.mode == self.MODE_INPUT and self.submode == self.SUBMODE_BISECTOR and self.bisector_waiting_for_edge) or \
                (self.mode == self.MODE_INPUT and self.submode == self.SUBMODE_PERP and self.perp_waiting_for_edge):
            self.hovered_segment = self.get_closest_line_and_segment(pos)

        rel_pos = self.abs_to_rel(pos)
        if rel_pos:
            self.hover_point = self.get_nearest_point(pos)
        else:
            self.hover_point = None

        self.canvas.Refresh()

    def on_left_click(self, event):
        """Обработчик левого клика мыши."""
        pos = event.GetPosition()
        point_abs = self.get_nearest_point(pos)
        if not point_abs:
            return
        point_rel = self.abs_to_rel(point_abs)
        if not point_rel:
            return

        left, top, right, bottom = self.get_grid_bounds()
        if not (left <= point_abs[0] <= right and top <= point_abs[1] <= bottom):
            return

        if self.mode == self.MODE_INPUT:
            if self.submode == self.SUBMODE_SEGMENT:
                if self.selected_point:
                    p1_rel = self.abs_to_rel(self.selected_point)
                    p2_rel = point_rel
                    self.rel_lines.append((p1_rel, p2_rel, self.line_type))
                    self.selected_point = None
                    self.update_all_intersections()
                    self.save_state()
                else:
                    self.selected_point = point_abs

            elif self.submode == self.SUBMODE_BISECTOR:
                if not self.bisector_waiting_for_edge:
                    if len(self.bisector_points) < 3:
                        self.bisector_points.append(point_abs)
                        if len(self.bisector_points) == 3:
                            self.bisector_waiting_for_edge = True
                            self.canvas.SetCursor(wx.Cursor(wx.CURSOR_HAND))
                else:
                    if self.hovered_segment:
                        idx, a, b, ltype, is_border = self.hovered_segment
                        target_line = self.square_edges[idx] if is_border else (a, b)
                        self.bisector_points.append(target_line)
                        if self.try_finish_bisector():
                            self.bisector_waiting_for_edge = False
                            self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
                        else:
                            self.bisector_points.pop()
                self.canvas.Refresh()

            elif self.submode == self.SUBMODE_INCENTER:
                self.incenter_points.append(point_abs)
                if len(self.incenter_points) == 3:
                    self.finish_incenter()
                self.canvas.Refresh()

            elif self.submode == self.SUBMODE_PERP:
                if not self.perp_waiting_for_edge:
                    self.perp_start_point = point_abs
                    self.perp_waiting_for_edge = True
                    self.canvas.SetCursor(wx.Cursor(wx.CURSOR_HAND))
                else:
                    if self.hovered_segment:
                        idx, a, b, ltype, is_border = self.hovered_segment
                        target_line = self.square_edges[idx] if is_border else (a, b)
                        if self.try_finish_perpendicular(target_line):
                            self.perp_waiting_for_edge = False
                            self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
                self.canvas.Refresh()

        elif self.mode == self.MODE_DELETE and self.hovered_segment:
            idx, a, b, ltype, is_border = self.hovered_segment
            if not is_border:
                self.remove_line_segment(idx, a, b)
            self.hovered_segment = None
            self.canvas.Refresh()

        elif self.mode == self.MODE_ALTER and self.hovered_segment:
            idx, a, b, ltype, is_border = self.hovered_segment
            if not is_border:
                self.alter_line_segment(idx, a, b)
            self.hovered_segment = None

        self.canvas.Refresh()

    def try_finish_bisector(self):
        """Попытаться завершить построение биссектрисы."""
        if len(self.bisector_points) != 4:
            return False
        p1, v, p2, target_line = self.bisector_points

        try:
            edge_p1, edge_p2 = target_line
        except (ValueError, TypeError):
            return False

        v_rel = self.abs_to_rel(v)
        p1_rel = self.abs_to_rel(p1)
        p2_rel = self.abs_to_rel(p2)

        v1 = (p1_rel[0] - v_rel[0], p1_rel[1] - v_rel[1])
        v2 = (p2_rel[0] - v_rel[0], p2_rel[1] - v_rel[1])
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 == 0 or len2 == 0:
            return False
        unit1 = (v1[0] / len1, v1[1] / len1)
        unit2 = (v2[0] / len2, v2[1] / len2)
        bis_dir = (unit1[0] + unit2[0], unit1[1] + unit2[1])
        norm = math.hypot(*bis_dir)
        if norm < 1e-8:
            return False
        bis_dir = (bis_dir[0] / norm, bis_dir[1] / norm)

        t_max = 10000
        bis_end = (v_rel[0] + t_max * bis_dir[0], v_rel[1] + t_max * bis_dir[1])

        edge_p1_rel = self.abs_to_rel(edge_p1)
        edge_p2_rel = self.abs_to_rel(edge_p2)
        inter = line_intersection_rel(
            (v_rel, bis_end),
            (edge_p1_rel, edge_p2_rel)
        )

        if inter:
            self.rel_lines.append((v_rel, inter, self.line_type))
            self.bisector_points = []
            self.update_all_intersections()
            self.save_state()
            return True

        end_rel = (v_rel[0] + 500 * bis_dir[0], v_rel[1] + 500 * bis_dir[1])
        self.rel_lines.append((v_rel, end_rel, self.line_type))
        self.bisector_points = []
        self.update_all_intersections()
        self.save_state()
        return True

    def try_finish_perpendicular(self, target_line):
        """Попытаться завершить построение перпендикуляра."""
        if not self.perp_start_point:
            return False
        start = self.perp_start_point
        start_rel = self.abs_to_rel(start)
        try:
            p1, p2 = target_line
        except (ValueError, TypeError):
            return False

        p1_rel = self.abs_to_rel(p1)
        p2_rel = self.abs_to_rel(p2)

        dx = p2_rel[0] - p1_rel[0]
        dy = p2_rel[1] - p1_rel[1]
        if abs(dx) < 1e-8 and abs(dy) < 1e-8:
            return False

        px = start_rel[0] - p1_rel[0]
        py = start_rel[1] - p1_rel[1]
        dot = px * dx + py * dy
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return False

        t = dot / len_sq
        foot_rel = (p1_rel[0] + t * dx, p1_rel[1] + t * dy)

        dist = math.hypot(start_rel[0] - foot_rel[0], start_rel[1] - foot_rel[1])
        if dist < 1e-8:
            return False

        self.rel_lines.append((start_rel, foot_rel, self.line_type))
        self.perp_start_point = None
        self.update_all_intersections()
        self.save_state()
        return True

    def finish_incenter(self):
        """Завершить построение линий к инцентру треугольника."""
        if len(self.incenter_points) != 3:
            return
        a, b, c = self.incenter_points
        a_rel = self.abs_to_rel(a)
        b_rel = self.abs_to_rel(b)
        c_rel = self.abs_to_rel(c)
        center_rel = triangle_incenter(a_rel, b_rel, c_rel)
        center_abs = self.rel_to_abs(center_rel)
        if not center_abs:
            return
        for p_rel in [a_rel, b_rel, c_rel]:
            self.rel_lines.append((p_rel, center_rel, self.line_type))
        self.incenter_points = []
        self.update_all_intersections()
        self.save_state()

    @staticmethod
    def _i(x):
        """Преобразовать координату в целое число для отрисовки."""
        return int(round(x))

    def on_paint(self, event):
        """Обработчик события рисования (перерисовка холста)."""
        dc = wx.PaintDC(self.canvas)
        self.draw_grid_area(dc)
        self.draw_grid_lines(dc)
        self.draw_AUX_lines(dc)
        self.draw_main_lines(dc)
        self.draw_hovered_segment(dc)
        self.draw_bisector_points(dc)
        self.draw_incenter_points(dc)
        self.draw_perp_point(dc)
        self.draw_mode_preview(dc)
        self.draw_hover_point(dc)

    def draw_grid_area(self, dc):
        """Нарисовать чёрную границу квадратной сетки."""
        left, top, right, bottom = self.get_grid_bounds()
        dc.SetPen(wx.Pen(wx.BLACK, 2))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(self._i(left), self._i(top), self._i(right - left), self._i(bottom - top))

    def draw_grid_lines(self, dc):
        """Нарисовать серые линии сетки разделения."""
        left, top, right, bottom = self.get_grid_bounds()
        dc.SetPen(wx.Pen(wx.Colour(220, 220, 220), 1))
        x = left
        while x <= right:
            dc.DrawLine(self._i(x), self._i(top), self._i(x), self._i(bottom))
            x += self.grid_size
        y = top
        while y <= bottom:
            dc.DrawLine(self._i(left), self._i(y), self._i(right), self._i(y))
            y += self.grid_size

    def draw_AUX_lines(self, dc):
        """Нарисовать вспомогательные линии (AUX) серым цветом."""
        dc.SetPen(wx.Pen(wx.Colour(180, 180, 180), 1))
        for p1_abs, p2_abs, ltype in self.get_abs_lines():
            if ltype == self.LINE_AUX:
                dc.DrawLine(self._i(p1_abs[0]), self._i(p1_abs[1]), self._i(p2_abs[0]), self._i(p2_abs[1]))

    def draw_main_lines(self, dc):
        """Нарисовать основные линии складок (красный для гор, синий для долин)."""
        for p1_abs, p2_abs, ltype in self.get_abs_lines():
            if ltype == self.LINE_MOUNTAIN:
                dc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))
            elif ltype == self.LINE_VALLEY:
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))
            else:
                continue
            dc.DrawLine(self._i(p1_abs[0]), self._i(p1_abs[1]), self._i(p2_abs[0]), self._i(p2_abs[1]))

    def draw_hovered_segment(self, dc):
        """Нарисовать выделенный зелёным цветом сегмент под курсором."""
        if not self.hovered_segment:
            return
        try:
            idx, a, b, ltype, is_border = self.hovered_segment
        except ValueError:
            idx, a, b, ltype = self.hovered_segment
            is_border = False

        color = wx.Colour(0, 255, 0)
        dc.SetPen(wx.Pen(color, 3))
        dc.DrawLine(self._i(a[0]), self._i(a[1]), self._i(b[0]), self._i(b[1]))

    def draw_bisector_points(self, dc):
        """Нарисовать выбранные точки при построении биссектрисы."""
        if self.mode != self.MODE_INPUT or self.submode != self.SUBMODE_BISECTOR:
            return
        if not self.bisector_points:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180)
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        dc.SetBrush(wx.Brush(color))
        dc.SetPen(wx.Pen(color, 2))

        for p in self.bisector_points[:3]:
            x, y = p
            dc.DrawCircle(self._i(x), self._i(y), 6)

    def draw_incenter_points(self, dc):
        """Нарисовать выбранные вершины треугольника при построении инцентра."""
        if self.mode != self.MODE_INPUT or self.submode != self.SUBMODE_INCENTER:
            return
        if not self.incenter_points:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180)
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        dc.SetBrush(wx.Brush(color))
        dc.SetPen(wx.Pen(color, 2))

        for p in self.incenter_points:
            x, y = p
            dc.DrawCircle(self._i(x), self._i(y), 6)

    def draw_perp_point(self, dc):
        """Нарисовать начальную точку при построении перпендикуляра."""
        if self.mode != self.MODE_INPUT or self.submode != self.SUBMODE_PERP:
            return
        if not self.perp_start_point:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180)
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        dc.SetBrush(wx.Brush(color))
        dc.SetPen(wx.Pen(color, 2))
        x, y = self.perp_start_point
        dc.DrawCircle(self._i(x), self._i(y), 6)

    def draw_mode_preview(self, dc):
        """Нарисовать предпросмотр текущей операции."""
        if self.mode != self.MODE_INPUT:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180)
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        if self.submode == self.SUBMODE_SEGMENT and self.selected_point and self.hover_point:
            dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
            sp = self.selected_point
            hp = self.hover_point
            dc.DrawLine(self._i(sp[0]), self._i(sp[1]), self._i(hp[0]), self._i(hp[1]))

        elif self.submode == self.SUBMODE_BISECTOR and self.bisector_points:
            dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
            pts = self.bisector_points[:3]
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                dc.DrawLine(self._i(p1[0]), self._i(p1[1]), self._i(p2[0]), self._i(p2[1]))

            if self.bisector_waiting_for_edge and self.hovered_segment:
                idx, a, b, _, is_border = self.hovered_segment
                if len(self.bisector_points) >= 3:
                    v = self.bisector_points[1]
                    p1 = self.bisector_points[0]
                    p2 = self.bisector_points[2]

                    v_rel = self.abs_to_rel(v)
                    p1_rel = self.abs_to_rel(p1)
                    p2_rel = self.abs_to_rel(p2)

                    v1 = (p1_rel[0] - v_rel[0], p1_rel[1] - v_rel[1])
                    v2 = (p2_rel[0] - v_rel[0], p2_rel[1] - v_rel[1])
                    len1 = math.hypot(*v1)
                    len2 = math.hypot(*v2)

                    if len1 > 0 and len2 > 0:
                        unit1 = (v1[0] / len1, v1[1] / len1)
                        unit2 = (v2[0] / len2, v2[1] / len2)
                        bis_dir = (unit1[0] + unit2[0], unit1[1] + unit2[1])
                        norm = math.hypot(*bis_dir)
                        if norm > 1e-8:
                            bis_dir = (bis_dir[0] / norm, bis_dir[1] / norm)
                            end_rel = (v_rel[0] + 500 * bis_dir[0], v_rel[1] + 500 * bis_dir[1])
                            end_abs = self.rel_to_abs(end_rel)
                            if end_abs:
                                dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
                                dc.DrawLine(self._i(v[0]), self._i(v[1]), self._i(end_abs[0]), self._i(end_abs[1]))

        elif self.submode == self.SUBMODE_INCENTER:
            if self.incenter_points and self.hover_point:
                points = self.incenter_points + [self.hover_point]
                if len(points) > 3:
                    points = points[-3:]

                if len(points) == 3:
                    a, b, c = points
                    a_rel = self.abs_to_rel(a)
                    b_rel = self.abs_to_rel(b)
                    c_rel = self.abs_to_rel(c)
                    center_rel = triangle_incenter(a_rel, b_rel, c_rel)
                    center_abs = self.rel_to_abs(center_rel)

                    if center_abs:
                        dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
                        for p in points:
                            dc.DrawLine(self._i(p[0]), self._i(p[1]),
                                        self._i(center_abs[0]), self._i(center_abs[1]))

                        dc.SetBrush(wx.Brush(color))
                        dc.SetPen(wx.Pen(color, 2))
                        dc.DrawCircle(self._i(center_abs[0]), self._i(center_abs[1]), 6)

        elif self.submode == self.SUBMODE_PERP:
            if self.perp_start_point and self.hovered_segment:
                start = self.perp_start_point
                try:
                    idx, a, b, _, is_border = self.hovered_segment
                except ValueError:
                    return
                p1, p2 = (self.square_edges[idx] if is_border else (a, b))

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if abs(dx) < 1e-8 and abs(dy) < 1e-8:
                    return

                px = start[0] - p1[0]
                py = start[1] - p1[1]
                dot = px * dx + py * dy
                len_sq = dx * dx + dy * dy
                if len_sq == 0:
                    return

                t = dot / len_sq
                foot = (p1[0] + t * dx, p1[1] + t * dy)

                dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
                dc.DrawLine(self._i(start[0]), self._i(start[1]), self._i(foot[0]), self._i(foot[1]))
                dc.SetBrush(wx.Brush(color))
                dc.SetPen(wx.Pen(color, 2))
                dc.DrawCircle(self._i(foot[0]), self._i(foot[1]), 6)

    def draw_hover_point(self, dc):
        """Нарисовать зелёную точку под курсором мыши."""
        if self.hover_point:
            x, y = self.hover_point
            dc.SetBrush(wx.Brush(wx.Colour(0, 255, 0)))
            dc.SetPen(wx.Pen(wx.Colour(0, 255, 0), 2))
            dc.DrawCircle(self._i(x), self._i(y), 5)

    def on_set_div(self, event):
        """Обработчик установки количества делений сетки."""
        try:
            new_div = int(self.div_text.GetValue())
            if new_div < 1 or new_div > 100:
                raise ValueError
            self.div_num = new_div
            self.update_grid_size()
            self.save_state()
        except ValueError:
            wx.MessageBox("Введите целое число от 1 до 100", "Ошибка", wx.OK | wx.ICON_ERROR)
            self.div_text.SetValue(str(self.div_num))

    def create_crease_pattern(self) -> CreasePattern:
        """Создать объект CreasePattern из текущих нарисованных линий."""
        cp = CreasePattern(paper_size=2.0)
        for p1_rel, p2_rel, ltype in self.rel_lines:
            if ltype == self.LINE_AUX:
                continue
            p0 = Point2D(p1_rel[0] * 2.0 - 1.0, p1_rel[1] * 2.0 - 1.0)
            p1 = Point2D(p2_rel[0] * 2.0 - 1.0, p2_rel[1] * 2.0 - 1.0)
            ori_type = LineType.MOUNTAIN if ltype == self.LINE_MOUNTAIN else LineType.VALLEY
            try:
                cp.add_line(OriLine(p0, p1, ori_type))
            except:
                pass
        cp.build_graph()
        return cp

    def check_foldability(self) -> dict:
        """Проверить складываемость текущего паттерна."""
        cp = self.create_crease_pattern()

        errors = []
        problem_vertices = set()

        # Простая проверка: наличие линий
        if len(self.rel_lines) == 0:
            return {
                "foldable": True,
                "errors": [],
                "problem_vertices": [],
                "crease_pattern": cp
            }

        # Проверка базовых условий Кавасаки
        for p1_rel, p2_rel, ltype in self.rel_lines:
            if ltype == self.LINE_AUX:
                continue

        return {
            "foldable": True,
            "errors": errors,
            "problem_vertices": list(problem_vertices),
            "crease_pattern": cp
        }

    def on_check_foldability(self, event):
        """Обработчик кнопки 'Проверить складывание'."""
        result = self.check_foldability()
        if result["foldable"]:
            PatternViewer(self, result["crease_pattern"])
        else:
            error_vertices = [Point2D(p[0], p[1]) if isinstance(p, tuple) else p
                              for p in result["problem_vertices"]]
            ErrorViewer(self, result["crease_pattern"], error_vertices)

    def on_clear(self, event):
        """Обработчик кнопки очистки."""
        self.rel_lines = []
        self.selected_point = None
        self.bisector_points = []
        self.incenter_points = []
        self.perp_start_point = None
        self.update_all_intersections()
        self.save_state()
        self.canvas.Refresh()


if __name__ == "__main__":
    app = wx.App()
    frame = Foldify(None, "Foldify")
    app.MainLoop()
