from __future__ import annotations
import wx
import math
from typing import Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Point2D:
    """Двумерная точка."""
    x: float
    y: float

    def __repr__(self) -> str:
        return f"P({self.x:.6f}, {self.y:.6f})"

    def distance_to(self, other: 'Point2D') -> float:
        """Вычислить расстояние до другой точки."""
        return math.hypot(self.x - other.x, self.y - other.y)


class LineType(Enum):
    """Типы линий."""
    MOUNTAIN = 1
    VALLEY = 2
    NONE = 0


class Foldify(wx.Frame):
    """Окно с квадратной областью сеткой и возможностью рисования."""

    MODE_INPUT = "input"
    MODE_DELETE = "delete"

    LINE_MOUNTAIN = "mountain"
    LINE_VALLEY = "valley"

    def __init__(self, parent, title):
        """Инициализировать главное окно приложения."""
        super(Foldify, self).__init__(parent, title=title, size=(900, 750),
                                      style=wx.DEFAULT_FRAME_STYLE | wx.RESIZE_BORDER)

        self.div_num = 4  # Количество делений сетки
        self.margin = 50
        self.scale = 1.0

        self.rel_lines = []  # Список линий в относительных координатах (-1 до 1)

        self.hover_point = None
        self.selected_point = None

        self.mode = self.MODE_INPUT
        self.line_type = self.LINE_MOUNTAIN

        self.square_edges = []
        self.grid_points = []

        # Основная панель
        self.panel = wx.Panel(self)
        self.canvas = wx.Panel(self.panel, style=wx.FULL_REPAINT_ON_RESIZE)
        self.canvas.SetBackgroundColour(wx.WHITE)

        # Контрольная панель
        self.control_panel = wx.Panel(self.panel)
        self.control_panel.SetBackgroundColour(wx.Colour(240, 240, 240))

        # Режимы работы
        self.radio_input = wx.RadioButton(self.control_panel, label="Ввод линии", style=wx.RB_GROUP)
        self.radio_delete = wx.RadioButton(self.control_panel, label="Удалить линию")

        self.radio_input.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_INPUT))
        self.radio_delete.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_DELETE))

        # Тип линии
        type_panel = wx.Panel(self.control_panel)
        type_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.radio_mountain = wx.RadioButton(type_panel, label="Гора", style=wx.RB_GROUP)
        self.radio_valley = wx.RadioButton(type_panel, label="Долина")
        self.radio_mountain.SetValue(True)
        type_sizer.Add(self.radio_mountain, 0, wx.ALL, 3)
        type_sizer.Add(self.radio_valley, 0, wx.ALL, 3)
        type_panel.SetSizer(type_sizer)

        self.radio_mountain.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_MOUNTAIN))
        self.radio_valley.Bind(wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_VALLEY))

        # Настройка сетки
        self.div_label = wx.StaticText(self.control_panel, label="Делений сетки:")
        self.div_text = wx.TextCtrl(self.control_panel, value="4", size=(50, -1), style=wx.TE_CENTER)
        self.set_button = wx.Button(self.control_panel, label="Задать")
        self.set_button.Bind(wx.EVT_BUTTON, self.on_set_div)

        # Кнопка очистки
        self.clear_button = wx.Button(self.control_panel, label="Очистить")
        self.clear_button.Bind(wx.EVT_BUTTON, self.on_clear)

        # Компоновка контрольной панели
        top_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(self.radio_input, 0, wx.ALL, 5)
        top_sizer.Add(self.radio_delete, 0, wx.ALL, 5)
        top_sizer.Add(wx.StaticLine(self.control_panel), 0, wx.EXPAND | wx.ALL, 5)

        type_sizer_main = wx.BoxSizer(wx.VERTICAL)
        type_sizer_main.Add(wx.StaticText(self.control_panel, label="Тип линии:"), 0, wx.ALL, 5)
        type_sizer_main.Add(type_panel, 0, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(type_sizer_main, 0, wx.EXPAND)
        top_sizer.Add(wx.StaticLine(self.control_panel), 0, wx.EXPAND | wx.ALL, 5)

        grid_sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer.Add(self.div_label, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        grid_sizer.Add(self.div_text, 0, wx.ALL | wx.EXPAND, 5)
        grid_sizer.Add(self.set_button, 0, wx.ALL | wx.EXPAND, 5)
        top_sizer.Add(grid_sizer, 0, wx.EXPAND)

        top_sizer.Add(wx.StaticLine(self.control_panel), 0, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(self.clear_button, 0, wx.ALL | wx.EXPAND, 5)
        top_sizer.AddStretchSpacer()

        self.control_panel.SetSizer(top_sizer)

        # Основная компоновка
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(self.control_panel, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 10)
        self.panel.SetSizer(main_sizer)

        # Обработчики событий
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_MOTION, self.on_mouse_move)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)

        # Инициализация сетки
        self.update_grid_size()
        self.Centre()
        self.Show()

    def set_main_mode(self, mode):
        """Установить главный режим работы."""
        self.mode = mode
        self.selected_point = None
        self.canvas.Refresh()

    def set_line_type(self, line_type):
        """Установить тип линии для рисования."""
        self.line_type = line_type

    def on_set_div(self, event):
        """Обработчик установки количества делений сетки."""
        try:
            new_div = int(self.div_text.GetValue())
            if new_div < 1 or new_div > 100:
                raise ValueError
            self.div_num = new_div
            self.update_grid_size()
        except ValueError:
            wx.MessageBox("Введите целое число от 1 до 100", "Ошибка", wx.OK | wx.ICON_ERROR)
            self.div_text.SetValue(str(self.div_num))

    def on_clear(self, event):
        """Обработчик очистки всех линий."""
        if wx.MessageBox("Вы уверены?", "Очистить чертёж", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            self.rel_lines = []
            self.selected_point = None
            self.hover_point = None
            self.canvas.Refresh()

    def update_grid_size(self):
        """Пересчитать параметры сетки."""
        self.canvas.Refresh()

    def get_grid_bounds(self) -> Tuple[float, float, float, float]:
        """Получить границы квадратной области (-1 до 1 в относительных координатах)."""
        return (-1.0, -1.0, 1.0, 1.0)

    def rel_to_abs(self, rel_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Конвертировать точку из относительных координат (-1 до 1) в абсолютные (пиксели)."""
        if not rel_point:
            return None
        w, h = self.canvas.GetSize()
        cx, cy = w // 2, h // 2
        scale = min(w, h) // 2 - self.margin
        x = cx + rel_point[0] * scale
        y = cy - rel_point[1] * scale
        return (x, y)

    def abs_to_rel(self, abs_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Конвертировать точку из абсолютных координат (пиксели) в относительные (-1 до 1)."""
        if not abs_point:
            return None
        w, h = self.canvas.GetSize()
        cx, cy = w // 2, h // 2
        scale = min(w, h) // 2 - self.margin
        if scale == 0:
            return None
        x = (abs_point[0] - cx) / scale
        y = (cy - abs_point[1]) / scale
        return (x, y)

    def get_nearest_point(self, pos: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Получить ближайшую точку сетки или на линиях к позиции мыши."""
        rel_pos = self.abs_to_rel(pos)
        if not rel_pos:
            return None

        # Проверка точек сетки
        snap_distance = 0.05
        for gp in self.grid_points:
            dist = math.hypot(rel_pos[0] - gp[0], rel_pos[1] - gp[1])
            if dist < snap_distance:
                return self.rel_to_abs(gp)

        # Проверка конечных точек линий
        for line in self.rel_lines:
            p0, p1, _ = line
            for p in [p0, p1]:
                dist = math.hypot(rel_pos[0] - p[0], rel_pos[1] - p[1])
                if dist < snap_distance:
                    return self.rel_to_abs(p)

        # Возвращаем исходную позицию, привязанную к сетке
        return self.rel_to_abs(rel_pos)

    def _i(self, val):
        """Конвертировать значение в целое число."""
        return int(val)

    def on_mouse_move(self, event):
        """Обработчик движения мыши."""
        pos = event.GetPosition()

        # Проверяем находимся ли внутри области рисования
        left, top, right, bottom = self.get_grid_bounds()
        rel_pos = self.abs_to_rel(pos)

        if rel_pos and (left <= rel_pos[0] <= right and top <= rel_pos[1] <= bottom):
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
        if not (left <= point_rel[0] <= right and top <= point_rel[1] <= bottom):
            return

        if self.mode == self.MODE_INPUT:
            if self.selected_point:
                # Завершить линию
                p1_rel = self.abs_to_rel(self.selected_point)
                p2_rel = point_rel
                self.rel_lines.append((p1_rel, p2_rel, self.line_type))
                self.selected_point = None
            else:
                # Начать новую линию
                self.selected_point = point_abs

        elif self.mode == self.MODE_DELETE:
            # Удалить ближайшую линию
            self._delete_nearest_line(point_rel)

        self.canvas.Refresh()

    def on_right_click(self, event):
        """Отменить текущий выбор при правом клике."""
        self.selected_point = None
        self.canvas.Refresh()

    def _delete_nearest_line(self, point_rel: Tuple[float, float]):
        """Удалить ближайшую линию к заданной точке."""
        min_dist = 0.1
        nearest_idx = None

        for i, (p0, p1, _) in enumerate(self.rel_lines):
            dist = self._point_to_line_distance(point_rel, p0, p1)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx is not None:
            self.rel_lines.pop(nearest_idx)

    @staticmethod
    def _point_to_line_distance(p: Tuple[float, float],
                                a: Tuple[float, float],
                                b: Tuple[float, float]) -> float:
        """Вычислить расстояние от точки до отрезка."""
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def on_paint(self, event):
        """Обработчик события рисования."""
        dc = wx.PaintDC(self.canvas)
        dc.Clear()
        dc.SetBackground(wx.Brush(wx.WHITE))
        dc.Clear()

        w, h = self.canvas.GetSize()
        cx, cy = w // 2, h // 2
        scale = min(w, h) // 2 - self.margin

        # Рисуем границу квадрата
        dc.SetPen(wx.Pen(wx.BLACK, 2))
        dc.SetBrush(wx.Brush(wx.WHITE))
        dc.DrawRectangle(
            self._i(cx - scale),
            self._i(cy - scale),
            self._i(2 * scale),
            self._i(2 * scale)
        )

        # Рисуем сетку
        self._draw_grid(dc, cx, cy, scale)

        # Рисуем линии
        self._draw_lines(dc, cx, cy, scale)

        # Рисуем точку которую держим
        if self.selected_point:
            dc.SetBrush(wx.Brush(wx.Colour(255, 200, 0)))
            dc.SetPen(wx.Pen(wx.Colour(255, 200, 0), 2))
            dc.DrawCircle(self._i(self.selected_point[0]), self._i(self.selected_point[1]), 5)

        # Рисуем зелёную точку под курсором
        if self.hover_point:
            dc.SetBrush(wx.Brush(wx.Colour(0, 255, 0)))
            dc.SetPen(wx.Pen(wx.Colour(0, 255, 0), 2))
            dc.DrawCircle(self._i(self.hover_point[0]), self._i(self.hover_point[1]), 5)

    def _draw_grid(self, dc, cx: float, cy: float, scale: float):
        """Рисовать сетку."""
        dc.SetPen(wx.Pen(wx.Colour(200, 200, 200), 1))

        # Создаём список точек сетки
        self.grid_points = []
        step = 2.0 / self.div_num

        for i in range(self.div_num + 1):
            for j in range(self.div_num + 1):
                x_rel = -1.0 + i * step
                y_rel = -1.0 + j * step
                self.grid_points.append((x_rel, y_rel))

                x_abs = cx + x_rel * scale
                y_abs = cy - y_rel * scale
                dc.DrawCircle(self._i(x_abs), self._i(y_abs), 2)

        # Рисуем горизонтальные и вертикальные линии
        for i in range(self.div_num + 1):
            pos = -1.0 + i * step

            # Вертикальная линия
            x1 = cx + pos * scale
            y1_top = cy + scale
            y1_bottom = cy - scale
            dc.DrawLine(self._i(x1), self._i(y1_top), self._i(x1), self._i(y1_bottom))

            # Горизонтальная линия
            y2 = cy - pos * scale
            x2_left = cx - scale
            x2_right = cx + scale
            dc.DrawLine(self._i(x2_left), self._i(y2), self._i(x2_right), self._i(y2))

    def _draw_lines(self, dc, cx: float, cy: float, scale: float):
        """Рисовать все линии."""
        for p0_rel, p1_rel, line_type in self.rel_lines:
            # Конвертируем в абсолютные координаты
            x0 = cx + p0_rel[0] * scale
            y0 = cy - p0_rel[1] * scale
            x1 = cx + p1_rel[0] * scale
            y1 = cy - p1_rel[1] * scale

            # Выбираем цвет в зависимости от типа линии
            if line_type == self.LINE_MOUNTAIN:
                dc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))
            elif line_type == self.LINE_VALLEY:
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))
            else:
                dc.SetPen(wx.Pen(wx.Colour(180, 180, 180), 2))

            dc.DrawLine(self._i(x0), self._i(y0), self._i(x1), self._i(y1))

        # Рисуем превью линии если начали рисовать
        if self.mode == self.MODE_INPUT and self.selected_point and self.hover_point:
            color = (wx.Colour(255, 0, 0) if self.line_type == self.LINE_MOUNTAIN else
                     wx.Colour(0, 0, 255))
            dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
            dc.DrawLine(self._i(self.selected_point[0]), self._i(self.selected_point[1]),
                        self._i(self.hover_point[0]), self._i(self.hover_point[1]))


if __name__ == "__main__":
    app = wx.App()
    frame = Foldify(None, "Foldify - Рисование сетки")
    app.MainLoop()