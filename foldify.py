from __future__ import annotations
import wx
import copy
import math
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import Enum, IntEnum
import itertools

EPS = 1e-10
EPS_AREA = 1e-8


def line_intersection_rel(line1, line2):
    """
    Найти точку пересечения двух линий, проверяя, находится ли пересечение
    внутри обоих отрезков.

    Args:
        line1 (tuple): Линия, представленная как ((x1, y1), (x2, y2)).
        line2 (tuple): Линия, представленная как ((x3, y3), (x4, y4)).

    Returns:
        tuple: Кортеж (x, y) координат пересечения или None, если пересечение
               не находится внутри обоих отрезков.
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


def point_to_line_distance(p, a, b):
    """
    Вычислить минимальное расстояние от точки до линейного отрезка.

    Args:
        p (tuple): Точка (x, y).
        a (tuple): Начало линии (x, y).
        b (tuple): Конец линии (x, y).

    Returns:
        float: Минимальное расстояние от точки до отрезка.
    """
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


def reflect_point_over_line(p, a, b):
    """
    Отразить точку относительно линии.

    Args:
        p (tuple): Точка для отражения (x, y).
        a (tuple): Начало линии (x, y).
        b (tuple): Конец линии (x, y).

    Returns:
        tuple: Отражённые координаты (x, y).
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return p
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    rx = 2 * proj_x - px
    ry = 2 * proj_y - py
    return (rx, ry)


def _merge_close_points(points: List[Point2D]) -> List[Point2D]:
    """
    Объединить точки, находящиеся близко друг к другу (в пределах EPS).

    Args:
        points (list): Список объектов Point2D.

    Returns:
        list: Список объектов Point2D с объединёнными близкими точками.
    """
    merged = []
    for p in points:
        if not any(p.is_close(q, EPS) for q in merged):
            merged.append(p)
    return merged


class LineType(Enum):
    """Перечисление типов линий в оригами паттерне.

    Attributes:
        MOUNTAIN: Линия горного сгиба (гора).
        VALLEY: Линия долинного сгиба (долина).
        AUX: Вспомогательная линия.
        CUT: Линия квадрата.
        NONE: Отсутствие типа.
    """

    MOUNTAIN = 1
    VALLEY = 2
    AUX = 3
    CUT = 4
    NONE = 0


class LayerRelation(IntEnum):
    """
    Перечисление для описания взаимного расположения слоёв при складывании.

    Attributes:
        ABOVE (int): Один слой находится на вершине другого.
        BELOW (int): Один слой находится под другим.
        UNKNOWN (int): неопределенное состояние
    """

    ABOVE = 1
    BELOW = -1
    UNKNOWN = 0


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

    def __add__(self, other: "Point2D") -> "Point2D":
        """
        Сложить две точки (как векторы).

        Args:
            other (Point2D): Вторая точка.

        Returns:
            Point2D: Результирующая точка.
        """
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2D") -> "Point2D":
        """
        Вычесть одну точку из другой (как векторы).

        Args:
            other (Point2D): Вторая точка.

        Returns:
            Point2D: Результирующая точка.
        """
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point2D":
        """
        Умножить точку на скаляр.

        Args:
            scalar (float): Множитель.

        Returns:
            Point2D: Результирующая точка.
        """
        return Point2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Point2D":
        """
        Разделить координаты точки на скаляр.

        Args:
            scalar (float): Делитель.

        Returns:
            Point2D: Результирующая точка.
        """
        return Point2D(self.x / scalar, self.y / scalar)

    def __repr__(self) -> str:
        """Возвращает строковое представление точки.

        Returns:
            Строка вида "P(x.xxxxxx, y.xxxxxx)".
        """
        return f"P({self.x:.6f}, {self.y:.6f})"

    def length(self) -> float:
        """
        Вычислить длину вектора от начала координат до точки.

        Returns:
            float: Длина вектора.
        """
        return math.hypot(self.x, self.y)

    def distance_to(self, other: "Point2D") -> float:
        """
        Вычислить расстояние до другой точки.

        Args:
            other (Point2D): Другая точка.

        Returns:
            float: Расстояние между точками.
        """
        return (self - other).length()

    def normalized(self) -> "Point2D":
        """
        Получить нормализованный вектор (единичный вектор).

        Returns:
            Point2D: Нормализованный вектор или нулевой вектор, если длина близка к нулю.
        """

        l = self.length()
        if l < 1e-10:
            return Point2D(0, 0)
        return self / l

    def dot(self, other: "Point2D") -> float:
        """Вычисляет скалярное произведение двух векторов.

        Args:
            other: Другой вектор.

        Returns:
            Скалярное произведение (float).
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Point2D") -> float:
        """
        Вычислить векторное произведение (Z-компонента) в 2D.

        Args:
            other (Point2D): Другой вектор.

        Returns:
            float: Z-компонента векторного произведения.
        """
        return self.x * other.y - self.y * other.x

    def perpendicular(self) -> "Point2D":
        """
        Получить перпендикулярный вектор (повёрнут на 90 градусов).

        Returns:
            Point2D: Перпендикулярный вектор.
        """
        return Point2D(-self.y, self.x)

    def rotated(self, angle_rad: float) -> "Point2D":
        """
        Получить вектор, повёрнутый на заданный угол.

        Args:
            angle_rad (float): Угол поворота в радианах.

        Returns:
            Point2D: Повёрнутый вектор.
        """
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return Point2D(self.x * c - self.y * s, self.x * s + self.y * c)

    def is_close(self, other: "Point2D", tol: float = 1e-10) -> bool:
        """
        Проверить, находится ли точка близко к другой точке.

        Args:
            other (Point2D): Другая точка.
            tol (float): Допуск расстояния. По умолчанию 1e-10.

        Returns:
            bool: True, если расстояние меньше допуска, иначе False.
        """
        return self.distance_to(other) < tol


@dataclass
class OriLine:
    """
    Ориентированная линия, представляющая отрезок с типом складки.

    Attributes:
        p0 (Point2D): Начальная точка.
        p1 (Point2D): Конечная точка.
        type (LineType): Тип линии (гора, долина, вспомогательная и т.д.).
    """

    p0: Point2D
    p1: Point2D
    type: LineType = LineType.AUX

    def __repr__(self) -> str:
        """Возвращает строковое представление линии.

        Returns:
            Строка вида "[T] точка1 — точка2", где T - первая буква типа.
        """
        t = {LineType.MOUNTAIN: "M", LineType.VALLEY: "V", LineType.AUX: "A"}.get(
            self.type, "?"
        )
        return f"[{t}] {self.p0} — {self.p1}"

    def __post_init__(self):
        """
        Проверить, что линия не является вырожденной.

        Raises:
            ValueError: Если начальная и конечная точки совпадают.
        """
        if self.p0.is_close(self.p1):
            raise ValueError("Degenerate line")

    def length(self) -> float:
        """
        Вычислить длину линии.

        Returns:
            float: Длина отрезка.
        """
        return self.p0.distance_to(self.p1)

    def direction(self) -> Point2D:
        """
        Получить нормализованный направляющий вектор линии.

        Returns:
            Point2D: Единичный вектор направления от p0 к p1.
        """
        return (self.p1 - self.p0).normalized()

    def midpoint(self) -> Point2D:
        """
        Получить середину линии.

        Returns:
            Point2D: Координаты средней точки.
        """
        return (self.p0 + self.p1) * 0.5

    def reversed(self) -> "OriLine":
        """
        Получить линию с обратным направлением.

        Returns:
            OriLine: Новая линия с переставленными p0 и p1.
        """
        return OriLine(self.p1, self.p0, self.type)

    def is_mountain(self) -> bool:
        """
        Получить линию с обратным направлением.

        Returns:
            OriLine: Новая линия с переставленными p0 и p1.
        """
        return self.type == LineType.MOUNTAIN

    def is_valley(self) -> bool:
        """
        Проверить, является ли линия долиной.

        Returns:
            bool: True, если тип линии VALLEY.
        """
        return self.type == LineType.VALLEY

    def is_fold_line(self) -> bool:
        """
        Проверить, является ли линия складкой (гора или долина).

        Returns:
            bool: True, если линия является складкой.
        """
        return self.type in (LineType.MOUNTAIN, LineType.VALLEY)


@dataclass
class OriVertex:
    """
    Вершина в графе паттерна складок.

    Attributes:
        p (Point2D): Координаты вершины.
        edges (list): Список рёбер, выходящих из этой вершины.
        pre_edges (list): Временный список для построения графа.
    """

    p: Point2D
    edges: List["OriEdge"] = field(default_factory=list, repr=False)
    pre_edges: List["OriEdge"] = field(default_factory=list, repr=False)

    def __hash__(self):
        """Возвращает хеш вершины на основе координат.

        Returns:
            Хеш координат точки.
        """
        return hash((self.p.x, self.p.y))

    def __eq__(self, other):
        """Проверяет равенство двух вершин по координатам.

        Args:
            other: Другая вершина для сравнения.

        Returns:
            True, если координаты вершин близки.
        """
        return isinstance(other, OriVertex) and self.p.is_close(other.p)

    def __repr__(self):
        """Возвращает строковое представление вершины.

        Returns:
            Строка вида "V(x.xxxxxx, y.xxxxxx)".
        """
        return f"V{self.p}"

    def __post_init__(self):
        """Инициализация листов после создания объекта."""
        self.edges = []
        self.pre_edges = []

    def add_edge(self, edge: "OriEdge"):
        """
        Добавить ребро к вершине.

        Args:
            edge (OriEdge): Ребро для добавления.
        """
        if edge not in self.pre_edges:
            self.pre_edges.append(edge)

    def degree(self) -> int:
        """
        Получить степень вершины (количество инцидентных рёбер).

        Returns:
            int: Количество рёбер.
        """
        return len(self.edges)


@dataclass
class OriEdge:
    """
    Ориентированное ребро в графе паттерна складок.

    Attributes:
        sv (OriVertex): Начальная вершина.
        ev (OriVertex): Конечная вершина.
        line (OriLine): Линия, которой принадлежит ребро.
        type (LineType): Тип ребра.
        opposite (OriEdge): Противоположное ребро.
        left_face (OriFace): Грань слева от ребра.
        right_face (OriFace): Грань справа от ребра.
    """

    sv: OriVertex
    ev: OriVertex
    line: OriLine
    type: LineType = LineType.AUX
    opposite: Optional["OriEdge"] = None
    left_face: Optional["OriFace"] = None
    right_face: Optional["OriFace"] = None

    def __eq__(self, other):
        """Проверяет равенство рёбер по идентификатору объекта.

        Args:
            other: Другое ребро для сравнения.

        Returns:
            True, если это один и тот же объект.
        """
        return isinstance(other, OriEdge) and id(self) == id(other)

    def __hash__(self):
        """Возвращает хеш на основе идентификатора объекта.

        Returns:
            Хеш объекта.
        """
        return id(self)

    def __repr__(self):
        """Возвращает строковое представление ребра.

        Returns:
            Строка вида "Edge(точка_начало → точка_конец, T)", где T - тип.
        """
        return f"Edge({self.sv.p} → {self.ev.p}, {self.type.name[0]})"

    def __post_init__(self):
        """
        Инициализировать тип ребра из типа линии.
        """
        self.type = self.line.type

    def is_fold_edge(self) -> bool:
        """
        Проверить, является ли ребро складкой (гора или долина).

        Returns:
            bool: True, если ребро является складкой.
        """
        return self.type in (LineType.MOUNTAIN, LineType.VALLEY)


@dataclass
class OriHalfedge:
    """
    Полурёбро (half-edge) в представлении двойного связного списка.

    Attributes:
        edge (OriEdge): Связанное ребро.
        face (OriFace): Грань, которой принадлежит полурёбро.
        vertex (OriVertex): Вершина в конце полурёбра.
        next (OriHalfedge): Следующее полурёбро в цикле грани.
        prev (OriHalfedge): Предыдущее полурёбро в цикле грани.
        opposite (OriHalfedge): Противоположное полурёбро.
    """

    edge: OriEdge
    face: Optional["OriFace"] = None
    vertex: OriVertex = None
    next: Optional["OriHalfedge"] = None
    prev: Optional["OriHalfedge"] = None
    opposite: Optional["OriHalfedge"] = None

    def __repr__(self):
        """Возвращает строковое представление полуребра.

        Returns:
            Строка вида "HE(точка_начало→точка_конец)".
        """
        return f"HE({self.edge.sv.p}→{self.edge.ev.p})"

    def __post_init__(self):
        """Инициализация после создания объекта.

        Устанавливает вершину на основе направления и ориентации грани.
        """
        if self.face is None:
            self.vertex = self.edge.ev
        else:
            self.vertex = self.edge.ev if self.face.is_ccw else self.edge.sv

    def direction_vector(self) -> Point2D:
        """
        Получить вектор направления полурёбра.

        Returns:
            Point2D: Вектор от начальной вершины к конечной.
        """
        return self.edge.ev.p - self.edge.sv.p


@dataclass
class OriFace:
    """
    Грань в паттерне складок, представляющая полигональную область.

    Attributes:
        halfedges (list): Список полурёбер, образующих контур грани.
        is_ccw (bool): Ориентирована ли грань против часовой стрелки.
        z_order (int): Порядок глубины для отображения.
        outline (list): Список точек контура грани.
    """

    halfedges: List[OriHalfedge] = field(default_factory=list)
    is_ccw: bool = True
    z_order: int = 0
    outline: List[Point2D] = field(default_factory=list, init=False)

    def __post_init__(self):
        """
        Инициализировать пустой контур.
        """
        self.outline = []

    def build_outline(self):
        """
        Построить контур грани из полурёбер.
        """
        if not self.halfedges:
            return
        self.outline = []
        he = self.halfedges[0]
        start_p = he.edge.sv.p if self.is_ccw else he.edge.ev.p
        current_p = start_p

        visited_ids = set()
        max_steps = len(self.halfedges) + 10  # защита от битых цепочек
        steps = 0

        while steps < max_steps:
            steps += 1
            self.outline.append(current_p)
            he_id = id(he)
            if he_id in visited_ids:
                break
            visited_ids.add(he_id)

            he = he.next
            if he is None:
                break

            current_p = he.edge.ev.p if self.is_ccw else he.edge.sv.p

            if current_p.is_close(start_p, 1e-8) and len(self.outline) >= 3:
                break

        if self.outline and not self.outline[-1].is_close(start_p, 1e-8):
            self.outline.append(start_p)

    def area(self) -> float:
        """
        Вычислить площадь грани.

        Returns:
            float: Площадь полигона.
        """
        if len(self.outline) < 3:
            return 0.0
        area = 0.0
        for i in range(len(self.outline)):
            p1 = self.outline[i]
            p2 = self.outline[(i + 1) % len(self.outline)]
            area += p1.x * p2.y - p2.x * p1.y
        return abs(area) * 0.5

    def center(self) -> Point2D:
        """
        Получить центр масс грани.

        Returns:
            Point2D: Координаты центра масс.
        """
        if not self.outline:
            return Point2D(0, 0)
        cx = sum(p.x for p in self.outline) / len(self.outline)
        cy = sum(p.y for p in self.outline) / len(self.outline)
        return Point2D(cx, cy)


class CreasePattern:
    """
    Паттерн складок, содержащий граф линий, вершин и рёбер.

    Attributes:
        paper_size (float): Размер квадратного листа бумаги.
        border_lines (list): Границы листа бумаги.
        lines (list): Все линии складок.
        vertices (list): Все вершины графа.
        edges (list): Все рёбра графа.
    """

    def __init__(self, paper_size: float = 2.0):
        """
        Инициализировать паттерн складок.

        Args:
            paper_size (float): Размер квадратного листа. По умолчанию 2.0.
        """
        self.paper_size = paper_size
        half = paper_size / 2.0
        self.border_lines = [
            OriLine(Point2D(-half, -half), Point2D(half, -half), LineType.AUX),
            OriLine(Point2D(half, -half), Point2D(half, half), LineType.AUX),
            OriLine(Point2D(half, half), Point2D(-half, half), LineType.AUX),
            OriLine(Point2D(-half, half), Point2D(-half, -half), LineType.AUX),
        ]
        self.lines: List[OriLine] = []
        self.vertices: List[OriVertex] = []
        self.edges: List[OriEdge] = []
        self._vertex_map: Dict[Point2D, OriVertex] = {}

    @staticmethod
    def _line_line_intersection(l1: OriLine, l2: OriLine) -> Optional[Point2D]:
        """
        Статический метод для поиска пересечения двух линий.

        Args:
            l1 (OriLine): Первая линия.
            l2 (OriLine): Вторая линия.

        Returns:
            Point2D: Точка пересечения или None.
        """
        a, b = l1.p0, l1.p1
        c, d = l2.p0, l2.p1

        denom = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x)
        if abs(denom) < EPS:
            return None

        t = ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)) / denom
        u = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)) / denom

        if 0.0 - EPS <= t <= 1.0 + EPS and 0.0 - EPS <= u <= 1.0 + EPS:
            ix = a.x + t * (b.x - a.x)
            iy = a.y + t * (b.y - a.y)
            return Point2D(ix, iy)
        return None

    def _find_all_intersections(self) -> Set[Point2D]:
        """
        Найти все точки пересечения между линиями и границами.

        Returns:
            set: Набор всех точек пересечения.
        """
        lines = self._all_lines_for_graph()
        points = set()

        for line in lines:
            points.add(line.p0)
            points.add(line.p1)

        for i, l1 in enumerate(lines):
            for l2 in lines[i + 1 :]:
                inter = self._line_line_intersection(l1, l2)
                if inter:
                    points.add(inter)

        return points

    def add_line(self, line: OriLine):
        """
        Добавить одну линию в паттерн.

        Args:
            line (OriLine): Линия для добавления.
        """
        if line.length() < EPS:
            return
        self.lines.append(line)

    def add_lines(self, lines: List[OriLine]):
        """
        Добавить список линий в паттерн.

        Args:
            lines (list): Список объектов OriLine.
        """
        for line in lines:
            self.add_line(line)

    def _all_lines_for_graph(self) -> List[OriLine]:
        """
        Получить все линии, включая границы, для построения графа.

        Returns:
            list: Объединённый список всех линий.
        """
        return self.lines + self.border_lines

    def _find_all_intersections(self) -> Set[Point2D]:
        lines = self._all_lines_for_graph()
        points = set()

        for line in lines:
            points.add(line.p0)
            points.add(line.p1)

        for i, l1 in enumerate(lines):
            for l2 in lines[i + 1 :]:
                inter = self._line_line_intersection(l1, l2)
                if inter:
                    points.add(inter)

        return points

    @staticmethod
    def _is_point_on_segment(p: Point2D, line: OriLine) -> bool:
        """
        Проверить, находится ли точка на отрезке линии.

        Args:
            p (Point2D): Точка для проверки.
            line (OriLine): Отрезок линии.

        Returns:
            bool: True, если точка на отрезке.
        """
        if p.is_close(line.p0, EPS) or p.is_close(line.p1, EPS):
            return True

        vec = line.p1 - line.p0
        if vec.length() < EPS:
            return False

        proj = (p - line.p0).dot(vec) / vec.dot(vec)
        if proj < -EPS or proj > (1.0 + EPS):
            return False

        foot = line.p0 + vec * proj
        return p.is_close(foot, EPS)

    def _split_line_at_points(
        self, line: OriLine, points_on_line: List[Point2D]
    ) -> List[OriLine]:
        """
        Разделить линию на отрезки в точках пересечения.

        Args:
            line (OriLine): Исходная линия.
            points_on_line (list): Точки на линии.

        Returns:
            list: Список разделённых отрезков.
        """
        if not points_on_line:
            return [line]

        vec = line.p1 - line.p0
        projections = []
        for p in points_on_line:
            t = (p - line.p0).dot(vec) / vec.dot(vec)
            projections.append((t, p))

        projections.sort(key=lambda x: x[0])

        interior = [pt for t, pt in projections if EPS < t < (1.0 - EPS)]
        sorted_points = [line.p0] + interior + [line.p1]

        unique = []
        for pt in sorted_points:
            if not unique or not unique[-1].is_close(pt, EPS):
                unique.append(pt)

        segments = []
        for i in range(len(unique) - 1):
            if unique[i].distance_to(unique[i + 1]) > EPS:
                segments.append(OriLine(unique[i], unique[i + 1], line.type))

        return segments

    def build_graph(self):
        """
        Построить граф вершин, рёбер и граней из линий паттерна.
        """
        self.vertices.clear()
        self.edges.clear()
        self._vertex_map.clear()

        all_points = self._find_all_intersections()
        all_points = _merge_close_points(list(all_points))

        for p in all_points:
            if self._is_inside_paper(p):
                v = OriVertex(p)
                self.vertices.append(v)
                self._vertex_map[p] = v

        split_segments: List[OriLine] = []
        for line in self._all_lines_for_graph():
            points_on_line = [
                p for p in all_points if self._is_point_on_segment(p, line)
            ]
            segments = self._split_line_at_points(line, points_on_line)
            split_segments.extend(segments)

        for seg in split_segments:
            v0 = self._vertex_map.get(seg.p0)
            v1 = self._vertex_map.get(seg.p1)
            if v0 is None or v1 is None:
                continue

            edge = OriEdge(sv=v0, ev=v1, line=seg)
            rev_edge = OriEdge(sv=v1, ev=v0, line=seg.reversed())

            edge.opposite = rev_edge
            rev_edge.opposite = edge

            self.edges.append(edge)
            self.edges.append(rev_edge)

            v0.add_edge(edge)
            v1.add_edge(edge)
            v0.add_edge(rev_edge)
            v1.add_edge(rev_edge)

        for v in self.vertices:
            v.edges = list(dict.fromkeys(v.pre_edges))
            v.pre_edges.clear()

    def _is_inside_paper(self, p: Point2D) -> bool:
        """
        Проверить, находится ли точка внутри листа бумаги.

        Args:
            p (Point2D): Точка для проверки.

        Returns:
            bool: True, если точка внутри листа.
        """
        half = self.paper_size / 2.0
        return -half - EPS <= p.x <= half + EPS and -half - EPS <= p.y <= half + EPS


class FaceBuilder:
    """
    Класс для построения граней из графа паттерна складок.

    Этот класс использует алгоритм обхода по левой рукой для нахождения
    максимальных граней в двойном связном списке.
    """

    @staticmethod
    def build_faces(cp: CreasePattern) -> List[OriFace]:
        """
        Построить список граней из паттерна складок.

        Args:
            cp (CreasePattern): Паттерн складок.

        Returns:
            list: Список объектов OriFace.
        """
        faces: List[OriFace] = []
        used_halfedges: Dict[OriEdge, bool] = {}

        def mark_used(edge: OriEdge):
            used_halfedges[edge] = True
            if edge.opposite:
                used_halfedges[edge.opposite] = True

        def is_used(edge: OriEdge) -> bool:
            return used_halfedges.get(edge, False)

        for edge in cp.edges:
            if edge.opposite:
                continue
            used_halfedges[edge] = False
            used_halfedges[edge.opposite] = False

        def follow_leftmost_turn(current_he: OriHalfedge) -> Optional[OriFace]:
            halfedges: List[OriHalfedge] = []
            start_edge = current_he.edge
            he = current_he

            while True:
                halfedges.append(he)
                mark_used(he.edge)
                candidates: List[OriEdge] = [
                    e for e in he.vertex.edges if not is_used(e)
                ]
                if not candidates:
                    break

                incoming_dir = (he.vertex.p - he.edge.sv.p) * -1.0
                best_edge: Optional[OriEdge] = None
                best_angle = -10.0

                for candidate in candidates:
                    outgoing_dir = (
                        candidate.ev.p - candidate.sv.p
                        if candidate.sv == he.vertex
                        else candidate.sv.p - candidate.ev.p
                    )
                    angle = math.atan2(
                        incoming_dir.cross(outgoing_dir), incoming_dir.dot(outgoing_dir)
                    )

                    if angle > best_angle:
                        best_angle = angle
                        best_edge = candidate

                if best_edge is None:
                    break

                next_vertex = (
                    best_edge.ev if best_edge.sv == he.vertex else best_edge.sv
                )
                he = OriHalfedge(
                    edge=best_edge, face=None, vertex=next_vertex  # заполним позже
                )

                if he.edge == start_edge and len(halfedges) > 2:
                    break

            if len(halfedges) < 3:
                for he in halfedges:
                    mark_used(he.edge)
                return None

            area_sum = 0.0
            for i in range(len(halfedges)):
                p1 = halfedges[i].vertex.p
                p2 = halfedges[(i + 1) % len(halfedges)].vertex.p
                area_sum += p1.x * p2.y - p2.x * p1.y

            is_ccw = area_sum > 0

            face = OriFace(halfedges=[], is_ccw=is_ccw)
            for i, he in enumerate(halfedges):
                correct_edge = (
                    he.edge
                    if (he.edge.sv == he.vertex) == (not is_ccw)
                    else he.edge.opposite
                )
                if correct_edge is None:
                    continue

                correct_next_vertex = (
                    correct_edge.ev
                    if correct_edge.sv.p.is_close(he.vertex.p)
                    else correct_edge.sv
                )

                proper_he = OriHalfedge(
                    edge=correct_edge, face=face, vertex=correct_next_vertex
                )
                face.halfedges.append(proper_he)

            for i in range(len(face.halfedges)):
                face.halfedges[i].next = face.halfedges[(i + 1) % len(face.halfedges)]
                face.halfedges[i].prev = face.halfedges[(i - 1) % len(face.halfedges)]

            for he in face.halfedges:
                if he.opposite is None and he.edge.opposite:
                    opp_vertex = (
                        he.edge.opposite.ev
                        if he.edge.opposite.sv == he.vertex
                        else he.edge.opposite.sv
                    )
                    he.opposite = OriHalfedge(
                        edge=he.edge.opposite, face=None, vertex=opp_vertex
                    )
                    he.opposite.opposite = he

            face.build_outline()
            if face.area() < EPS_AREA:
                return None

            return face

        for edge in cp.edges:
            if is_used(edge) or edge.opposite is None:
                continue

            for start_dir in [edge, edge.opposite]:
                if is_used(start_dir):
                    continue
                he = OriHalfedge(edge=start_dir, face=None, vertex=start_dir.ev)
                face = follow_leftmost_turn(he)
                if face:
                    for he in face.halfedges:
                        he.face = face
                        if he.opposite:
                            he.opposite.face = face  # временно, может быть перезаписано
                    faces.append(face)

        unique_faces = []
        seen_outlines = set()
        for face in faces:
            outline_key = tuple((p.x, p.y) for p in face.outline)
            if outline_key not in seen_outlines and face.area() > EPS_AREA:
                seen_outlines.add(outline_key)
                unique_faces.append(face)

        return unique_faces


class FoldabilityChecker:
    """
    Класс для проверки возможности складывания паттерна складок.

    Проверяет корректность паттерна складок согласно правилам оригами,
    включая проверку углов и условия Маекавы и Кавасаки.
    """

    @staticmethod
    def check_local_flat_foldability(
        cp: CreasePattern, faces: List[OriFace]
    ) -> List[str]:
        """
        Проверить локальную возможность плоского складывания в каждой вершине.

        Проверяет условия Маекавы (|M - V| = 2) и Кавасаки (суммы углов = 180°)
        для каждой внутренней вершины паттерна.

        Args:
            cp (CreasePattern): Паттерн складок для проверки.
            faces (list): Список граней паттерна.

        Returns:
            list: Список строк с описанием найденных ошибок. Пустой список если ошибок нет.
        """
        errors = []
        half = cp.paper_size / 2.0

        for vertex in cp.vertices:
            x, y = vertex.p.x, vertex.p.y
            if (abs(abs(x) - half) < EPS) or (abs(abs(y) - half) < EPS):
                continue

            fold_edges = [e for e in vertex.edges if e.is_fold_edge()]
            if len(fold_edges) < 2:
                continue

            rays = []
            for edge in fold_edges:
                if edge.sv.p.is_close(vertex.p):
                    dir_vec = edge.ev.p - vertex.p
                else:
                    dir_vec = edge.sv.p - vertex.p
                angle = math.atan2(dir_vec.y, dir_vec.x)
                rays.append((edge.type, angle))

            rays.sort(key=lambda x: x[1])

            unique_rays = []
            for typ, ang in rays:
                if (
                    not unique_rays
                    or abs((ang - unique_rays[-1][1] + math.pi * 2) % (math.pi * 2))
                    > 1e-8
                ):
                    unique_rays.append((typ, ang))

            if len(unique_rays) < 2:
                continue

            n = len(unique_rays)
            if n % 2 != 0:
                errors.append(f"Вершина {vertex.p}: нечётное количество складок ({n})")
                continue

            m_count = sum(1 for t, _ in unique_rays if t == LineType.MOUNTAIN)
            v_count = sum(1 for t, _ in unique_rays if t == LineType.VALLEY)
            if abs(m_count - v_count) != 2:
                errors.append(
                    f"Вершина {vertex.p}: Maekawa нарушено |M-V| = {abs(m_count - v_count)}"
                )

            angles = []
            for i in range(n):
                a1 = unique_rays[i][1]
                a2 = unique_rays[(i + 1) % n][1]
                diff = (a2 - a1 + math.pi * 2) % (math.pi * 2)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                angles.append(diff)

            sum_even = sum(angles[i] for i in range(0, n, 2))
            sum_odd = sum(angles[i] for i in range(1, n, 2))

            if abs(math.degrees(sum_even) - 180.0) > 2.0:
                errors.append(
                    f"Вершина {vertex.p}: Kawasaki (чётные): {math.degrees(sum_even):.1f}° ≈ 180°"
                )
            if abs(math.degrees(sum_odd) - 180.0) > 2.0:
                errors.append(
                    f"Вершина {vertex.p}: Kawasaki (нечётные): {math.degrees(sum_odd):.1f}° ≈ 180°"
                )

        return errors

    @staticmethod
    def check_global_conditions(faces: List[OriFace]) -> List[str]:
        """
        Проверить глобальные условия складываемости.

        Проверяет условия, связанные с общей структурой паттерна,
        а не с отдельными вершинами.

        Args:
            faces (list): Список граней паттерна.

        Returns:
            list: Список строк с описанием найденных глобальных ошибок.
        """

        return []

    @staticmethod
    def full_foldability_check(
        cp: CreasePattern, faces: List[OriFace]
    ) -> Dict[str, Any]:
        """
        Выполнить полную проверку складываемости паттерна.

        Объединяет локальные и глобальные проверки и возвращает итоговый результат.

        Args:
            cp (CreasePattern): Паттерн складок для проверки.
            faces (list): Список граней паттерна.

        Returns:
            dict: Словарь с ключами:
                  - "foldable" (bool): Может ли паттерн быть сложен плоско.
                  - "errors" (list): Список найденных ошибок.
                  - "face_count" (int): Количество граней.
                  - "vertex_count" (int): Количество вершин.
        """
        local_errors = FoldabilityChecker.check_local_flat_foldability(cp, faces)

        return {
            "foldable": len(local_errors) == 0,
            "errors": local_errors,
            "face_count": len(faces),
            "vertex_count": len(cp.vertices),
        }


class FoldedModel:
    """
    Модель сложенного оригами, содержащая информацию о трансформированных гранях.

    Attributes:
        faces (list): Список граней с учётом трансформаций при складывании.
        overlaps_resolved (bool): Флаг, указывающий, был ли разрешен порядок наложения слоёв.
        error (str): Сообщение об ошибке, если порядок слоёв не может быть разрешен.
    """

    def __init__(self):
        """
        Инициализировать пустую модель сложенного оригами.
        """
        self.faces: List[OriFace] = []
        self.overlaps_resolved = False
        self.error: Optional[str] = None


class StackingAnalyzer:
    """
    Класс для анализа и определения порядка наложения слоёв при складывании.

    Использует информацию о типах складок (гора/долина) для определения,
    какие грани находятся выше других после складывания.
    """

    @staticmethod
    def estimate_layer_order(cp: CreasePattern, faces: List[OriFace]) -> FoldedModel:
        """
        Определить порядок наложения слоёв на основе паттерна складок.

        Использует граничные условия (складка гора = грань сверху, долина = грань снизу)
        и логический вывод для определения полного порядка слоёв.

        Args:
            cp (CreasePattern): Паттерн складок.
            faces (list): Список граней паттерна.

        Returns:
            FoldedModel: Модель с определённым порядком слоёв (z_order для каждой грани).
                        Если порядок не может быть определён, возвращает модель с ошибкой.
        """
        model = FoldedModel()
        model.faces = copy.deepcopy(faces)

        if not faces:
            model.error = "Нет граней"
            return model

        n = len(model.faces)
        relations: List[List[LayerRelation]] = [
            [LayerRelation.UNKNOWN] * n for _ in range(n)
        ]

        for face1_idx, face1 in enumerate(model.faces):
            for he in face1.halfedges:
                edge = he.edge
                if not edge.is_fold_edge():
                    continue
                opp_he = he.opposite
                if opp_he is None or opp_he.face is None:
                    continue

                face2 = opp_he.face
                face2_idx = next(i for i, f in enumerate(model.faces) if f is face2)

                if edge.type == LineType.MOUNTAIN:
                    relations[face1_idx][face2_idx] = LayerRelation.ABOVE
                    relations[face2_idx][face1_idx] = LayerRelation.BELOW
                elif edge.type == LineType.VALLEY:
                    relations[face1_idx][face2_idx] = LayerRelation.BELOW
                    relations[face2_idx][face1_idx] = LayerRelation.ABOVE

        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(n):
                    if i == j or relations[i][j] == LayerRelation.UNKNOWN:
                        continue
                    for k in range(n):
                        if relations[j][k] == LayerRelation.UNKNOWN:
                            continue

                        expected = relations[i][j] * relations[j][k]
                        if expected == 0:
                            continue

                        new_rel = LayerRelation(1 if expected > 0 else -1)

                        if relations[i][k] == LayerRelation.UNKNOWN:
                            relations[i][k] = new_rel
                            relations[k][i] = LayerRelation(-new_rel.value)
                            changed = True
                        elif relations[i][k] != new_rel:
                            model.error = (
                                f"Конфликт порядка слоёв между гранями {i} и {k}"
                            )
                            return model

        z_values = [0] * n
        used_z = set()

        def assign_z(idx: int, z: int) -> bool:
            """
            Рекурсивно назначить значение z-порядка грани и всем связанным граням.

            Args:
                idx (int): Индекс текущей грани в массиве граней.
                z (int): Значение z-порядка для назначения этой грани.

            Returns:
                bool: True, если z-порядок успешно назначен для всей цепочки граней.
                      False, если обнаружен конфликт (противоречие в отношениях слоёв).
            """
            if z_values[idx] != 0 and z_values[idx] != z:
                return False
            if z_values[idx] == z:
                return True
            if z in used_z:
                return False

            z_values[idx] = z
            used_z.add(z)

            for other_idx in range(n):
                rel = relations[idx][other_idx]
                if rel == LayerRelation.ABOVE:
                    if not assign_z(other_idx, z - 1):
                        return False
                elif rel == LayerRelation.BELOW:
                    if not assign_z(other_idx, z + 1):
                        return False
            return True

        largest_idx = max(range(n), key=lambda i: model.faces[i].area())
        if not assign_z(largest_idx, 0):
            model.error = "Не удалось разрешить порядок слоёв"
            return model

        if all(z == 0 for z in z_values):
            min_z = 0
        else:
            min_z = min(z for z in z_values if z != 0)

        for i, z in enumerate(z_values):
            model.faces[i].z_order = z - min_z

        model.overlaps_resolved = True
        return model


class GeometryUtil:
    """
    Утилита для геометрических вычислений и преобразований.

    Содержит статические методы для отражения точек и поворота относительно центра.
    """

    @staticmethod
    def reflect_point_over_line(
        point: Point2D, line_p0: Point2D, line_p1: Point2D
    ) -> Point2D:
        """
        Отразить точку относительно линии.

        Вычисляет проекцию точки на линию, а затем отражает точку
        относительно этой проекции.

        Args:
            point (Point2D): Точка для отражения.
            line_p0 (Point2D): Начало линии отражения.
            line_p1 (Point2D): Конец линии отражения.

        Returns:
            Point2D: Отражённая точка.
        """
        px, py = point.x, point.y
        x1, y1 = line_p0.x, line_p0.y
        x2, y2 = line_p1.x, line_p1.y

        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < EPS and abs(dy) < EPS:
            return point

        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        rx = 2 * proj_x - px
        ry = 2 * proj_y - py
        return Point2D(rx, ry)

    @staticmethod
    def rotate_point_180(center: Point2D, point: Point2D) -> Point2D:
        """
        Повернуть точку на 180 градусов относительно центра.

        Эквивалентно отражению точки относительно центра.

        Args:
            center (Point2D): Центр поворота.
            point (Point2D): Точка для поворота.

        Returns:
            Point2D: Повёрнутая точка.
        """
        return center + (center - point)


class FoldedModelBuilder:
    """
    Класс для преобразования паттерна складок в трёхмерную модель.
    """

    @staticmethod
    def build_folded_model(cp: CreasePattern, model: FoldedModel) -> List[OriFace]:
        """
        Построить трёхмерную модель из паттерна складок и порядка слоёв.

        Трансформирует контуры граней в соответствии с типами складок (гора/долина).
        Для гор граница отражается относительно линии складки.
        Для долин граница отражается и затем поворачивается на 180°.

        Args:
            cp (CreasePattern): Паттерн складок.
            model (FoldedModel): Модель со скрытым порядком слоёв.

        Returns:
            list: Список трансформированных граней (OriFace), отсортированных по z_order.
        """
        if not model.faces:
            return []

        outlines = {}
        transformed = set()

        for face in model.faces:
            outlines[id(face)] = list(face.outline)

        for orig_face in model.faces:
            face_id = id(orig_face)

            if face_id in transformed:
                continue

            for he in orig_face.halfedges:
                edge = he.edge

                if not edge.is_fold_edge():
                    continue

                opp_he = he.opposite
                if not opp_he or not opp_he.face:
                    continue

                neighbor_face = opp_he.face
                neighbor_z = neighbor_face.z_order
                my_z = orig_face.z_order

                # Если сосед находится НИЖЕ текущей грани
                if neighbor_z < my_z:
                    # Трансформируем текущую грань ЭТО И ТОЛЬКО ЭТО
                    p0 = edge.sv.p
                    p1 = edge.ev.p

                    new_outline = [
                        GeometryUtil.reflect_point_over_line(p, p0, p1)
                        for p in outlines[face_id]
                    ]

                    if edge.type == LineType.VALLEY:
                        center = edge.line.midpoint()
                        new_outline = [
                            GeometryUtil.rotate_point_180(center, p)
                            for p in new_outline
                        ]

                    outlines[face_id] = new_outline
                    transformed.add(face_id)
                    break

        result_faces = []
        for face in model.faces:
            face_copy = copy.deepcopy(face)
            face_copy.outline = outlines[id(face)]
            face_copy.build_outline()
            result_faces.append(face_copy)

        result_faces.sort(key=lambda f: f.z_order)
        return result_faces


class OrigamiFolder:
    """
    Основной класс для полного процесса складывания паттерна оригами.

    Координирует построение граней, проверку складываемости, анализ порядка
    слоёв и построение трёхмерной модели.
    """

    @staticmethod
    def fold_crease_pattern(cp: CreasePattern) -> Dict[str, Any]:
        """
        Выполнить полный процесс складывания паттерна.

        Последовательно:
        1. Строит грани из паттерна
        2. Проверяет условия Маекавы и Кавасаки
        3. Определяет порядок наложения слоёв
        4. Трансформирует грани в трёхмерную модель

        Args:
            cp (CreasePattern): Паттерн складок для складывания.

        Returns:
            dict: Словарь с ключами:
                  - "success" (bool): Успешно ли выполнено складывание.
                  - "folded_faces" (list): Список трансформированных граней (если успешно).
                  - "original_faces" (list): Исходные грани до трансформации.
                  - "crease_pattern" (CreasePattern): Использованный паттерн.
                  - "errors" (list): Список ошибок (если неудачно).
        """
        faces = FaceBuilder.build_faces(cp)
        if not faces:
            return {
                "success": False,
                "errors": ["Не удалось построить грани (возможно, топология нарушена)"],
            }

        check_result = FoldabilityChecker.full_foldability_check(cp, faces)
        if not check_result["foldable"]:
            return {"success": False, "errors": check_result["errors"]}

        stacked_model = StackingAnalyzer.estimate_layer_order(cp, faces)
        if stacked_model.error:
            return {"success": False, "errors": [stacked_model.error]}

        folded_faces = FoldedModelBuilder.build_folded_model(cp, stacked_model)

        return {
            "success": True,
            "folded_faces": folded_faces,
            "original_faces": faces,
            "crease_pattern": cp,
        }


class FoldViewer(wx.Frame):
    """
    Окно для трёхмерной визуализации сложенного оригами.

    Отображает слои модели с различными оттенками в зависимости от z_order.
    Более высокие слои имеют более интенсивный цвет.

    Attributes:
        folded_faces (list): Список граней модели для отображения.
        panel (wx.Panel): Панель для рисования.
    """

    def __init__(self, parent, folded_faces):
        """
        Инициализировать окно просмотра сложенной модели.

        Args:
            parent: Родительское окно.
            folded_faces (list): Список граней OriFace для отображения.
        """
        super().__init__(parent, title="Сложенная модель", size=(800, 800))
        self.folded_faces = folded_faces
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.Centre()
        self.Show()

    def on_paint(self, event):
        """
        Обработчик события рисования (перерисовка модели).

        Отрисовывает все грани модели с сортировкой по z_order,
        используя полупрозрачные полигоны для изображения слоёв.

        Args:
            event: Событие рисования wxPython.
        """
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
    """
    Окно для визуализации паттерна складок с отмеченными проблемными вершинами.

    Показывает паттерн складок и красным цветом выделяет вершины,
    нарушающие условия Маекавы или Кавасаки.

    Attributes:
        crease_pattern (CreasePattern): Паттерн для визуализации.
        problem_vertices (list): Список проблемных вершин (Point2D).
        panel (wx.Panel): Панель для рисования.
    """

    def __init__(self, parent, crease_pattern, problem_vertices):
        """
        Инициализировать окно визуализации ошибок.

        Args:
            parent: Родительское окно.
            crease_pattern (CreasePattern): Паттерн для отображения.
            problem_vertices (list): Список проблемных вершин (Point2D).
        """
        super().__init__(
            parent, title="Проверка складывания - Обнаружены ошибки", size=(800, 800)
        )
        self.crease_pattern = crease_pattern
        self.problem_vertices = problem_vertices
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.Centre()
        self.Show()

    def on_paint(self, event):
        """
        Обработчик события рисования (перерисовка паттерна с ошибками).

        Рисует:
        - Границу квадрата чёрным цветом
        - Линии складок (красный для гор, синий для долин)
        - Все вершины чёрными точками
        - Проблемные вершины красными кружками

        Args:
            event: Событие рисования wxPython.
        """
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
        dc.DrawRectangle(
            int(cx - scale), int(cy - scale), int(2 * scale), int(2 * scale)
        )

        for line in self.crease_pattern.lines:
            p0 = line.p0
            p1 = line.p1
            x0 = cx + p0.x * scale
            y0 = cy - p0.y * scale
            x1 = cx + p1.x * scale
            y1 = cy - p1.y * scale

            if line.type == LineType.MOUNTAIN:
                dc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))  # Красный для гор
            elif line.type == LineType.VALLEY:
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))  # Синий для долин
            else:
                dc.SetPen(
                    wx.Pen(wx.Colour(180, 180, 180), 1)
                )  # Серый для вспомогательных

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


class Foldify(wx.Frame):
    """
    Главное окно приложения для редактирования и анализа паттернов оригами.

    Предоставляет полный интерфейс для:
    - Рисования паттернов складок (горы, долины, вспомогательные линии)
    - Использования геометрических инструментов (биссектриса, инцентр, перпендикуляр)
    - Проверки складываемости
    - Складывания и просмотра результатов
    - Импорта и экспорта паттернов в формате JSON

    Режимы работы:
    - MODE_INPUT: Рисование новых линий
    - MODE_DELETE: Удаление существующих сегментов
    - MODE_ALTER: Изменение типа линии (гора <-> долина)

    Подрежимы (только в MODE_INPUT):
    - SUBMODE_SEGMENT: Рисование отрезков
    - SUBMODE_BISECTOR: Построение биссектрисы угла
    - SUBMODE_INCENTER: Построение линий к инцентру треугольника
    - SUBMODE_PERP: Построение перпендикуляра к линии

    Attributes:
        canvas (wx.Panel): Холст для рисования паттерна.
        rel_lines (list): Список линий в относительных координатах.
        rel_endpoint_points (list): Конечные точки линий (кроме AUX).
        rel_intersections (set): Пересечения между складками.
        rel_invisible (set): Пересечения со вспомогательными линиями.
        mode (str): Текущий режим работы.
        submode (str): Текущий подрежим (при MODE_INPUT).
        line_type (str): Тип линии для рисования.
        history (list): История состояний для undo/redo.
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

    def __init__(self, parent, title, s="src/foldify"):
        """
        Инициализировать главное окно приложения.

        Создаёт интерфейс с холстом для рисования и панелью управления.
        Устанавливает обработчики событий для мыши и клавиатуры.

        Args:
            parent: Родительское окно.
            title (str): Заголовок окна.
            s (str): Директория с иконками для кнопок инструментов. По умолчанию "src/foldify".
        """
        super(Foldify, self).__init__(
            parent,
            title=title,
            size=(900, 650),
            style=wx.DEFAULT_FRAME_STYLE | wx.RESIZE_BORDER,
        )

        self.div_num = 4
        self.margin = 50
        self.scale = 1.0

        self.rel_lines = []

        self.rel_endpoint_points = []
        self.rel_intersections = set()
        self.rel_invisible = set()

        self.hover_point = None
        self.selected_point = None

        self.mode = self.MODE_INPUT
        self.submode = self.SUBMODE_SEGMENT
        self.line_type = self.LINE_MOUNTAIN
        self.hovered_line = None
        self.hovered_segment = None

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

        self.radio_input = wx.RadioButton(
            self.control_panel, label="Ввод линии", style=wx.RB_GROUP
        )
        self.radio_delete = wx.RadioButton(self.control_panel, label="Удалить линию")
        self.radio_alter = wx.RadioButton(self.control_panel, label="Изменить тип")

        self.radio_input.Bind(
            wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_INPUT)
        )
        self.radio_delete.Bind(
            wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_DELETE)
        )
        self.radio_alter.Bind(
            wx.EVT_RADIOBUTTON, lambda e: self.set_main_mode(self.MODE_ALTER)
        )

        type_panel = wx.Panel(self.control_panel)
        type_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.radio_mountain = wx.RadioButton(
            type_panel, label="Гора", style=wx.RB_GROUP
        )
        self.radio_valley = wx.RadioButton(type_panel, label="Долина")
        self.radio_aux = wx.RadioButton(type_panel, label="Вспомогательная")
        self.radio_mountain.SetValue(True)
        type_sizer.Add(self.radio_mountain, 0, wx.ALL, 3)
        type_sizer.Add(self.radio_valley, 0, wx.ALL, 3)
        type_sizer.Add(self.radio_aux, 0, wx.ALL, 3)
        type_panel.SetSizer(type_sizer)

        self.radio_mountain.Bind(
            wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_MOUNTAIN)
        )
        self.radio_valley.Bind(
            wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_VALLEY)
        )
        self.radio_aux.Bind(
            wx.EVT_RADIOBUTTON, lambda e: self.set_line_type(self.LINE_AUX)
        )

        icon_dir = s
        self.icons = {
            "segment": [
                self.load_icon(os.path.join(icon_dir, "segment_inactive.png")),
                self.load_icon(os.path.join(icon_dir, "segment_active.png")),
            ],
            "bisector": [
                self.load_icon(os.path.join(icon_dir, "bisector_inactive.png")),
                self.load_icon(os.path.join(icon_dir, "bisector_active.png")),
            ],
            "incenter": [
                self.load_icon(os.path.join(icon_dir, "incenter_inactive.png")),
                self.load_icon(os.path.join(icon_dir, "incenter_active.png")),
            ],
            "perp": [
                self.load_icon(os.path.join(icon_dir, "perp_inactive.png")),
                self.load_icon(os.path.join(icon_dir, "perp_active.png")),
            ],
        }

        self.btn_segment = wx.BitmapButton(
            self.control_panel, bitmap=self.icons["segment"][1], size=(32, 32)
        )
        self.btn_bisector = wx.BitmapButton(
            self.control_panel, bitmap=self.icons["bisector"][0], size=(32, 32)
        )
        self.btn_incenter = wx.BitmapButton(
            self.control_panel, bitmap=self.icons["incenter"][0], size=(32, 32)
        )
        self.btn_perp = wx.BitmapButton(
            self.control_panel, bitmap=self.icons["perp"][0], size=(32, 32)
        )

        self.btn_segment.SetToolTip("Отрезок")
        self.btn_bisector.SetToolTip("Биссектриса")
        self.btn_incenter.SetToolTip("Центр вписанной окружности")
        self.btn_perp.SetToolTip("Перпендикуляр")

        self.btn_segment.Bind(
            wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_SEGMENT)
        )
        self.btn_bisector.Bind(
            wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_BISECTOR)
        )
        self.btn_incenter.Bind(
            wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_INCENTER)
        )
        self.btn_perp.Bind(wx.EVT_BUTTON, lambda e: self.set_submode(self.SUBMODE_PERP))

        self.div_label = wx.StaticText(self.control_panel, label="Размер сетки")
        self.div_text = wx.TextCtrl(
            self.control_panel, value="4", size=(50, -1), style=wx.TE_CENTER
        )
        self.set_button = wx.Button(self.control_panel, label="Задать")
        self.set_button.Bind(wx.EVT_BUTTON, self.on_set_div)

        self.undo_button = wx.Button(self.control_panel, label="Назад")
        self.redo_button = wx.Button(self.control_panel, label="Вперед")
        self.check_button = wx.Button(self.control_panel, label="Проверить складывание")

        self.fold_button = wx.Button(self.control_panel, label="Сложить")
        self.fold_button.Bind(wx.EVT_BUTTON, self.on_fold)

        self.undo_button.Bind(wx.EVT_BUTTON, self.on_undo)
        self.redo_button.Bind(wx.EVT_BUTTON, self.on_redo)
        self.check_button.Bind(wx.EVT_BUTTON, self.on_check_foldability)

        top_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(self.radio_input, 0, wx.ALL, 5)
        top_sizer.Add(self.radio_delete, 0, wx.ALL, 5)
        top_sizer.Add(self.radio_alter, 0, wx.ALL, 5)
        top_sizer.Add(type_panel, 0, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(wx.StaticLine(self.control_panel), 0, wx.EXPAND | wx.ALL, 5)

        icon_row = wx.BoxSizer(wx.HORIZONTAL)
        icon_row.Add(self.btn_segment, 0, wx.ALL, 8)
        icon_row.Add(self.btn_bisector, 0, wx.ALL, 8)
        icon_row.Add(self.btn_incenter, 0, wx.ALL, 8)
        icon_row.Add(self.btn_perp, 0, wx.ALL, 8)

        bottom_sizer = wx.BoxSizer(wx.VERTICAL)
        bottom_sizer.Add(self.div_label, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        bottom_sizer.Add(self.div_text, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.set_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.AddStretchSpacer()
        bottom_sizer.Add(self.undo_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.redo_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.check_button, 0, wx.ALL | wx.EXPAND, 5)
        bottom_sizer.Add(self.fold_button, 0, wx.ALL | wx.EXPAND, 5)

        ctrl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctrl_sizer.Add(top_sizer, 0, wx.EXPAND)
        ctrl_sizer.Add(icon_row, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        ctrl_sizer.Add(bottom_sizer, 1, wx.EXPAND)
        self.control_panel.SetSizer(ctrl_sizer)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(self.control_panel, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(self.canvas, 1, wx.EXPAND)
        self.panel.SetSizer(main_sizer)

        menubar = wx.MenuBar()
        file_menu = wx.Menu()
        export_item = file_menu.Append(wx.ID_SAVEAS, "Экспорт...\tCtrl+S")
        import_item = file_menu.Append(wx.ID_OPEN, "Импорт...\tCtrl+O")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "Выход")
        menubar.Append(file_menu, "&Файл")
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.on_export, export_item)
        self.Bind(wx.EVT_MENU, self.on_import, import_item)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), exit_item)

        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_MOTION, self.on_mouse_move)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.canvas.Bind(wx.EVT_SIZE, self.on_resize)
        self.canvas.Bind(wx.EVT_KEY_DOWN, self.on_key_down)

        self.update_grid_size()
        self.save_state()
        self.update_undo_redo_buttons()
        self.update_all_buttons()
        self.Show()

    def create_crease_pattern(self) -> CreasePattern:
        """
        Создать объект CreasePattern из текущих нарисованных линий.

        Преобразует относительные координаты в абсолютные (-1 до 1),
        исключая вспомогательные линии.

        Returns:
            CreasePattern: Построенный паттерн складок.
        """
        cp = CreasePattern(paper_size=2.0)  # от -1 до +1, как в ORIPA
        for p1_rel, p2_rel, ltype in self.rel_lines:
            if ltype == self.LINE_AUX:
                continue
            p0 = Point2D(p1_rel[0] * 2.0 - 1.0, p1_rel[1] * 2.0 - 1.0)
            p1 = Point2D(p2_rel[0] * 2.0 - 1.0, p2_rel[1] * 2.0 - 1.0)
            ori_type = (
                LineType.MOUNTAIN if ltype == self.LINE_MOUNTAIN else LineType.VALLEY
            )
            cp.add_line(OriLine(p0, p1, ori_type))

        cp.build_graph()
        return cp

    def on_fold(self, event):
        """
        Обработчик кнопки "Сложить".

        Выполняет полный процесс складывания:
        1. Проверяет складываемость
        2. Если ошибки - показывает их в ErrorViewer
        3. Если OK - построить модель и показать в FoldViewer

        Args:
            event: Событие кнопки wxPython.
        """

        # Сначала выполняем проверку складываемости
        check_result = self.check_foldability()

        if not check_result["foldable"]:
            # Показываем окно с визуализацией ошибок вместо MessageBox
            error_vertices = [
                Point2D(p[0], p[1]) if isinstance(p, tuple) else p
                for p in check_result["problem_vertices"]
            ]
            ErrorViewer(self, check_result["crease_pattern"], error_vertices)
            return

        # Если проверка пройдена, продолжаем со складыванием
        cp = self.create_crease_pattern()
        result = OrigamiFolder.fold_crease_pattern(cp)

        if not result["success"]:
            # Этот блок теперь должен срабатывать редко, только при других ошибках
            error_text = "\n".join(f"• {e}" for e in result["errors"])
            wx.MessageBox(
                f"Возникли проблемы при складывании:\n\n{error_text}",
                "Ошибка складывания",
                wx.OK | wx.ICON_ERROR,
            )
            return

        folded_faces = result["folded_faces"]

        if len(self.rel_lines) == 0 or all(
            ltype == self.LINE_AUX for _, _, ltype in self.rel_lines
        ):
            single_face = OriFace()
            half = 1.0
            single_face.outline = [
                Point2D(-half, -half),
                Point2D(half, -half),
                Point2D(half, half),
                Point2D(-half, half),
            ]
            single_face.z_order = 0
            FoldViewer(self, [single_face])
        else:
            FoldViewer(self, folded_faces)

    def load_icon(self, path):
        """
        Загрузить иконку из файла или создать пустую иконку.

        Args:
            path (str): Путь к файлу иконки PNG.

        Returns:
            wx.Bitmap: Загруженная или пустая иконка.
        """
        if not os.path.exists(path):
            img = wx.Image(32, 32)
            img.Clear(200)
            return wx.Bitmap(img)
        img = wx.Image(path, wx.BITMAP_TYPE_PNG)
        return wx.Bitmap(img)

    def set_main_mode(self, mode):
        """
        Переключиться на основной режим работы.

        Args:
            mode (str): Новый режим (MODE_INPUT, MODE_DELETE, MODE_ALTER).
        """
        self.mode = mode
        if mode == self.MODE_INPUT:
            self.submode = self.SUBMODE_SEGMENT
        self.reset_drawing_state()
        self.update_all_buttons()
        self.canvas.Refresh()

    def set_submode(self, submode):
        """
        Переключиться на подрежим в режиме ввода.

        Только работает если текущий режим MODE_INPUT.

        Args:
            submode (str): Новый подрежим.
        """
        if self.mode != self.MODE_INPUT:
            return
        self.submode = submode
        self.reset_drawing_state()
        self.update_submode_buttons()
        self.canvas.Refresh()

    def reset_drawing_state(self):
        """
        Очистить состояние рисования (выбранные точки, ожидание и т.д.).
        """
        self.selected_point = None
        selfistector_points = []
        self.bisector_waiting_for_edge = False
        self.incenter_points = []
        self.perp_start_point = None
        self.perp_waiting_for_edge = False
        self.hovered_line = None
        self.hovered_segment = None
        self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

    def update_all_buttons(self):
        """
        Обновить состояние кнопок управления согласно текущему режиму.
        """
        self.radio_input.SetValue(self.mode == self.MODE_INPUT)
        self.radio_delete.SetValue(self.mode == self.MODE_DELETE)
        self.radio_alter.SetValue(self.mode == self.MODE_ALTER)

        self.btn_segment.Enable(self.mode == self.MODE_INPUT)
        self.btn_bisector.Enable(self.mode == self.MODE_INPUT)
        self.btn_incenter.Enable(self.mode == self.MODE_INPUT)
        self.btn_perp.Enable(self.mode == self.MODE_INPUT)

        if self.mode == self.MODE_INPUT:
            self.update_submode_buttons()

    def update_submode_buttons(self):
        """
        Обновить визуальное состояние кнопок инструментов подрежимов.
        """
        mapping = {
            self.SUBMODE_SEGMENT: self.btn_segment,
            self.SUBMODE_BISECTOR: self.btn_bisector,
            self.SUBMODE_INCENTER: self.btn_incenter,
            self.SUBMODE_PERP: self.btn_perp,
        }
        for sub, btn in mapping.items():
            key = {
                self.btn_segment: "segment",
                self.btn_bisector: "bisector",
                self.btn_incenter: "incenter",
                self.btn_perp: "perp",
            }[btn]
            idx = 1 if self.submode == sub else 0
            btn.SetBitmap(self.icons[key][idx])

    def set_line_type(self, line_type):
        """
        Установить тип линии для рисования.

        Args:
            line_type (str): Новый тип (LINE_MOUNTAIN, LINE_VALLEY, LINE_AUX).
        """
        self.line_type = line_type
        self.canvas.Refresh()

    def save_state(self):
        """
        Сохранить текущее состояние в историю для undo/redo.
        """
        state = {"div_num": self.div_num, "rel_lines": copy.deepcopy(self.rel_lines)}
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        self.history.append(state)
        self.history_index += 1
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
        self.update_undo_redo_buttons()

    def restore_state(self, state):
        """
        Восстановить состояние из истории.

        Args:
            state (dict): Сохранённое состояние.
        """
        self.div_num = state["div_num"]
        self.div_text.SetValue(str(self.div_num))
        self.rel_lines = copy.deepcopy(state["rel_lines"])
        self.update_grid_size()
        self.update_all_intersections()
        self.update_endpoint_points()
        self.canvas.Refresh()

    def on_undo(self, event):
        """
        Обработчик отмены последнего действия (Ctrl+Z).

        Args:
            event: Событие меню/кнопки wxPython.
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.restore_state(self.history[self.history_index])
            self.update_undo_redo_buttons()

    def on_redo(self, event):
        """
        Обработчик повтора отменённого действия (Ctrl+Y).

        Args:
            event: Событие меню/кнопки wxPython.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.restore_state(self.history[self.history_index])
            self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        """
        Обновить состояние кнопок Undo/Redo согласно истории.
        """
        self.undo_button.Enable(self.history_index > 0)
        self.redo_button.Enable(self.history_index < len(self.history) - 1)

    def on_key_down(self, event):
        """
        Обработчик нажатия клавиши.

        Args:
            event: Событие клавиши wxPython.
        """
        key = event.GetKeyCode()
        if event.ControlDown():
            if key == ord("Z"):
                self.on_undo(None)
            elif key == ord("Y"):
                self.on_redo(None)
            elif key == ord("S"):
                self.on_export(None)
            elif key == ord("O"):
                self.on_import(None)
        event.Skip()

    def update_grid_size(self):
        """
        Пересчитать размер сетки при изменении размера окна или количества делений.
        """
        w, h = self.canvas.GetSize()
        if w <= 0 or h <= 0:
            return

        usable = min(w, h)
        self.scale = (
            usable / (usable - 2 * self.margin) if usable > 2 * self.margin else 1.0
        )
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
            ((left, top), (left, bottom)),
        ]

        self._grid_bounds = (left, top, right, bottom)

        self.update_all_intersections()
        self.canvas.Refresh()

    def get_grid_bounds(self):
        """
        Получить границы сетки в абсолютных координатах экрана.

        Returns:
            tuple: (left, top, right, bottom) в пиксельных координатах.
        """
        return self._grid_bounds

    def rel_to_abs(self, rel_point):
        """
        Преобразовать точку из относительных координат (0-1) в абсолютные (экран).

        Args:
            rel_point (tuple): Точка в относительных координатах (x, y).

        Returns:
            tuple: Точка в абсолютных координатах (x, y) или None если вне сетки.
        """
        if not rel_point:
            return None
        left, top, right, bottom = self.get_grid_bounds()
        size = right - left
        x_rel, y_rel = rel_point
        return (left + x_rel * size, bottom - y_rel * size)

    def abs_to_rel(self, abs_point):
        """
        Преобразовать точку из абсолютных (экран) в относительные координаты (0-1).

        Args:
            abs_point (tuple): Точка в абсолютных координатах (x, y).

        Returns:
            tuple: Точка в относительных координатах (x, y) или None если вне сетки.
        """
        if not abs_point:
            return None
        left, top, right, bottom = self.get_grid_bounds()
        size = right - left
        if size == 0:
            return None
        x_abs, y_abs = abs_point
        return ((x_abs - left) / size, (bottom - y_abs) / size)

    def on_resize(self, event):
        """
        Обработчик изменения размера окна.

        Args:
            event: Событие изменения размера wxPython.
        """
        wx.CallAfter(self.update_grid_size)
        event.Skip()

    def get_abs_lines(self):
        """
        Получить все линии в абсолютных координатах экрана.

        Returns:
            list: Список кортежей (p1_abs, p2_abs, type) для каждой линии.
        """
        return [
            (self.rel_to_abs(p1), self.rel_to_abs(p2), ltype)
            for p1, p2, ltype in self.rel_lines
        ]

    def get_nearest_point(self, pos):
        """
        Найти ближайшую значимую точку к позиции курсора.

        Проверяет в порядке приоритета:
        1. Точки сетки
        2. Конечные точки линий
        3. Пересечения складок
        4. Пересечения вспомогательных линий

        Args:
            pos (tuple): Позиция в абсолютных координатах (x, y).

        Returns:
            tuple: Ближайшая точка (x, y) или None.
        """
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

        candidates = (
            [grid_point]
            + self.rel_endpoint_points
            + list(self.rel_intersections)
            + list(self.rel_invisible)
        )

        candidates = list(set(candidates))
        candidates_abs = [self.rel_to_abs(p) for p in candidates if self.rel_to_abs(p)]

        if not candidates_abs:
            return None

        return min(
            candidates_abs, key=lambda p: math.hypot(p[0] - pos[0], p[1] - pos[1])
        )

    def update_all_intersections(self):
        """
        Пересчитать все пересечения между линиями и сеткой.

        Обновляет rel_intersections и rel_invisible.
        """
        self.rel_intersections = set()
        self.rel_invisible = set()

        mv_lines = [
            (p1, p2, ltype)
            for p1, p2, ltype in self.rel_lines
            if ltype != self.LINE_AUX
        ]
        for i in range(len(mv_lines)):
            for j in range(i + 1, len(mv_lines)):
                inter = line_intersection_rel(mv_lines[i][:2], mv_lines[j][:2])
                if inter:
                    self.rel_intersections.add(inter)

        aux_lines = [
            (p1, p2) for p1, p2, ltype in self.rel_lines if ltype == self.LINE_AUX
        ]

        for k in range(1, self.div_num):
            y_rel = k / self.div_num
            x_rel = k / self.div_num
            grid_h = ((0.0, y_rel), (1.0, y_rel))
            grid_v = ((x_rel, 0.0), (x_rel, 1.0))
            for aux in aux_lines:
                inter = line_intersection_rel(aux, grid_h)
                if inter:
                    self.rel_invisible.add(inter)
                inter = line_intersection_rel(aux, grid_v)
                if inter:
                    self.rel_invisible.add(inter)

        bounds = [
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (0, 1)),
            ((0, 1), (0, 0)),
        ]
        for b1, b2 in bounds:
            for aux in aux_lines:
                inter = line_intersection_rel(aux, (b1, b2))
                if inter:
                    self.rel_invisible.add(inter)

        for aux in aux_lines:
            for mv_p1, mv_p2, _ in mv_lines:
                inter = line_intersection_rel(aux, (mv_p1, mv_p2))
                if inter:
                    self.rel_invisible.add(inter)

        for i in range(len(aux_lines)):
            for j in range(i + 1, len(aux_lines)):
                inter = line_intersection_rel(aux_lines[i], aux_lines[j])
                if inter:
                    self.rel_invisible.add(inter)

    def update_endpoint_points(self):
        """
        Пересчитать список конечных точек всех складок (кроме AUX).
        """
        self.rel_endpoint_points = []
        for p1_rel, p2_rel, ltype in self.rel_lines:
            if ltype != self.LINE_AUX:
                self.rel_endpoint_points.append(p1_rel)
                self.rel_endpoint_points.append(p2_rel)
        self.rel_endpoint_points = list(set(self.rel_endpoint_points))

    def get_closest_line_and_segment(self, pos):
        """
        Найти ближайший сегмент линии к позиции курсора.

        Args:
            pos (tuple): Позиция в абсолютных координатах (x, y).

        Returns:
            tuple: Кортеж (index, start_point, end_point, line_type, is_border) или None.
                   start_point и end_point - это точки, ограничивающие сегмент.
        """
        x, y = pos
        min_dist = float("inf")
        result = None

        for idx, (p1_rel, p2_rel, ltype) in enumerate(self.rel_lines):
            p1 = self.rel_to_abs(p1_rel)
            p2 = self.rel_to_abs(p2_rel)
            points_on_line = [p1, p2]
            for inter_rel in self.rel_intersections:
                inter_abs = self.rel_to_abs(inter_rel)
                if inter_abs and self.is_point_on_segment(
                    inter_abs, (p1, p2), tol=1e-3
                ):
                    points_on_line.append(inter_abs)
            for inter_rel in self.rel_invisible:
                inter_abs = self.rel_to_abs(inter_rel)
                if inter_abs and self.is_point_on_segment(
                    inter_abs, (p1, p2), tol=1e-3
                ):
                    points_on_line.append(inter_abs)

            points_on_line = sorted(set(points_on_line), key=lambda pt: (pt[0], pt[1]))

            for i in range(len(points_on_line) - 1):
                a, b = points_on_line[i], points_on_line[i + 1]
                dist = self.distance_to_segment(pos, (a, b))
                if dist < 20 and dist < min_dist:
                    min_dist = dist
                    result = (idx, a, b, ltype, False)

        for edge_idx, (p1, p2) in enumerate(self.square_edges):
            points_on_edge = [p1, p2]
            for inter_rel in self.rel_intersections:
                inter_abs = self.rel_to_abs(inter_rel)
                if inter_abs and self.is_point_on_segment(
                    inter_abs, (p1, p2), tol=1e-3
                ):
                    points_on_edge.append(inter_abs)
            for inter_rel in self.rel_invisible:
                inter_abs = self.rel_to_abs(inter_rel)
                if inter_abs and self.is_point_on_segment(
                    inter_abs, (p1, p2), tol=1e-3
                ):
                    points_on_edge.append(inter_abs)

            points_on_edge = sorted(set(points_on_edge), key=lambda pt: (pt[0], pt[1]))

            for i in range(len(points_on_edge) - 1):
                a, b = points_on_edge[i], points_on_edge[i + 1]
                dist = self.distance_to_segment(pos, (a, b))
                if dist < 20 and dist < min_dist:
                    min_dist = dist
                    result = (edge_idx, a, b, "border", True)

        return result

    def distance_to_segment(self, point, segment):
        """
        Вычислить минимальное расстояние от точки до отрезка.

        Args:
            point (tuple): Точка (x, y).
            segment (tuple): Отрезок ((x1, y1), (x2, y2)).

        Returns:
            float: Минимальное расстояние.
        """
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
        """
        Проверить, находится ли точка на отрезке (в пределах допуска).

        Args:
            point (tuple): Точка (x, y).
            segment (tuple): Отрезок ((x1, y1), (x2, y2)).
            tol (float): Допуск расстояния.

        Returns:
            bool: True если точка на отрезке.
        """
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
        """
        Проверить, совпадают ли две точки в пределах допуска.

        Args:
            p1 (tuple): Первая точка (x, y).
            p2 (tuple): Вторая точка (x, y).
            tol (float): Допуск расстояния.

        Returns:
            bool: True если точки совпадают.
        """
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    def remove_line_segment(self, line_idx, seg_start, seg_end):
        """
        Удалить определённый сегмент линии.

        Если линия содержит несколько сегментов (разделена пересечениями),
        удаляет только указанный сегмент, сохраняя остальные.

        Args:
            line_idx (int): Индекс линии в rel_lines.
            seg_start (tuple): Начало сегмента для удаления.
            seg_end (tuple): Конец сегмента для удаления.
        """
        p1_rel, p2_rel, ltype = self.rel_lines[line_idx]
        p1 = self.rel_to_abs(p1_rel)
        p2 = self.rel_to_abs(p2_rel)
        points_on_line = [p1, p2]
        for inter_rel in self.rel_intersections:
            inter_abs = self.rel_to_abs(inter_rel)
            if self.is_point_on_segment(inter_abs, (p1, p2), tol=1e-3):
                points_on_line.append(inter_abs)
        points_on_line = sorted(set(points_on_line), key=lambda pt: (pt[0], pt[1]))

        start_idx = next(
            (
                i
                for i, pt in enumerate(points_on_line)
                if self.points_equal(pt, seg_start)
            ),
            -1,
        )
        end_idx = next(
            (
                i
                for i, pt in enumerate(points_on_line)
                if self.points_equal(pt, seg_end)
            ),
            -1,
        )

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

        self.update_endpoint_points()
        self.update_all_intersections()
        self.save_state()

    def alter_line_segment(self, line_idx, seg_start, seg_end):
        """
        Изменить тип определённого сегмента линии (гора <-> долина).

        Только применяется к складкам (не для AUX).

        Args:
            line_idx (int): Индекс линии в rel_lines.
            seg_start (tuple): Начало сегмента для изменения.
            seg_end (tuple): Конец сегмента для изменения.
        """
        p1_rel, p2_rel, ltype = self.rel_lines[line_idx]
        p1 = self.rel_to_abs(p1_rel)
        p2 = self.rel_to_abs(p2_rel)
        points_on_line = [p1, p2]
        for inter_rel in self.rel_intersections:
            inter_abs = self.rel_to_abs(inter_rel)
            if self.is_point_on_segment(inter_abs, (p1, p2), tol=1e-3):
                points_on_line.append(inter_abs)
        points_on_line = sorted(set(points_on_line), key=lambda pt: (pt[0], pt[1]))

        start_idx = next(
            (
                i
                for i, pt in enumerate(points_on_line)
                if self.points_equal(pt, seg_start)
            ),
            -1,
        )
        end_idx = next(
            (
                i
                for i, pt in enumerate(points_on_line)
                if self.points_equal(pt, seg_end)
            ),
            -1,
        )

        if start_idx == -1 or end_idx == -1:
            return

        new_type = (
            self.LINE_VALLEY if ltype == self.LINE_MOUNTAIN else self.LINE_MOUNTAIN
        )

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

        self.update_endpoint_points()
        self.update_all_intersections()
        self.save_state()

    def _get_last_selected_point(self):
        """
        Получить последнюю выбранную точку согласно текущему подрежиму.

        Returns:
            tuple: Координаты (x, y) или None если точки не выбраны.
        """
        if self.submode == self.SUBMODE_SEGMENT:
            return self.selected_point
        elif self.submode == self.SUBMODE_BISECTOR:
            return self.bisector_points[-1] if self.bisector_points else None
        elif self.submode == self.SUBMODE_INCENTER:
            return self.incenter_points[-1] if self.incenter_points else None
        elif self.submode == self.SUBMODE_PERP:
            return self.perp_start_point
        return None

    def _is_point_selected(self, point):
        """
        Проверить, выбрана ли точка в текущем подрежиме.

        Args:
            point (tuple): Точка для проверки (x, y).

        Returns:
            bool: True если точка выбрана.
        """
        if self.submode == self.SUBMODE_SEGMENT:
            return self.points_equal(self.selected_point, point)
        elif self.submode == self.SUBMODE_BISECTOR:
            return any(self.points_equal(p, point) for p in self.bisector_points)
        elif self.submode == self.SUBMODE_INCENTER:
            return any(self.points_equal(p, point) for p in self.incenter_points)
        elif self.submode == self.SUBMODE_PERP:
            return self.points_equal(self.perp_start_point, point)
        return False

    def _reset_selection_from_index(self, index):
        """
        Сбросить выделение начиная с указанного индекса (для многошаговых операций).

        Args:
            index (int): Индекс, с которого сбросить выделение.
        """
        if self.submode == self.SUBMODE_BISECTOR:
            self.bisector_points = self.bisector_points[:index]
            self.bisector_waiting_for_edge = len(self.bisector_points) == 3
        elif self.submode == self.SUBMODE_INCENTER:
            self.incenter_points = self.incenter_points[:index]
        self.canvas.Refresh()

    def _handle_reselection(self, point_abs):
        """
        Обработать переклик по уже выбранной точке (отмена выбора).

        Args:
            point_abs (tuple): Точка в абсолютных координатах (x, y).

        Returns:
            bool: True если переклик был обработан.
        """
        if self.submode == self.SUBMODE_SEGMENT:
            if self.selected_point and self.points_equal(
                self.selected_point, point_abs
            ):
                self.selected_point = None
                self.canvas.Refresh()
                return True

        elif self.submode == self.SUBMODE_BISECTOR:
            for i, p in enumerate(self.bisector_points):
                if self.points_equal(p, point_abs):
                    self._reset_selection_from_index(i)
                    return True

        elif self.submode == self.SUBMODE_INCENTER:
            for i, p in enumerate(self.incenter_points):
                if self.points_equal(p, point_abs):
                    self._reset_selection_from_index(i)
                    return True

        elif self.submode == self.SUBMODE_PERP:
            if self.perp_start_point and self.points_equal(
                self.perp_start_point, point_abs
            ):
                self.perp_start_point = None
                self.perp_waiting_for_edge = False
                self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
                self.canvas.Refresh()
                return True

        return False

    def on_mouse_move(self, event):
        """
        Обработчик движения мыши.

        Обновляет позицию наведения и ищет близкие линии (в режимах удаления/изменения).

        Args:
            event: Событие движения мыши wxPython.
        """
        pos = event.GetPosition()

        if (
            self.mode in [self.MODE_DELETE, self.MODE_ALTER]
            or (
                self.mode == self.MODE_INPUT
                and self.submode == self.SUBMODE_BISECTOR
                and self.bisector_waiting_for_edge
            )
            or (
                self.mode == self.MODE_INPUT
                and self.submode == self.SUBMODE_PERP
                and self.perp_waiting_for_edge
            )
        ):

            self.hover_point = None
            self.hovered_line = None
            self.hovered_segment = None
            result = self.get_closest_line_and_segment(pos)
            if result:
                self.hovered_segment = result
        else:
            self.hover_point = self.get_nearest_point(pos)
            self.hovered_line = None
            self.hovered_segment = None

        self.canvas.Refresh()

    def on_left_click(self, event):
        """
        Обработчик левого клика мыши.

        Обработка зависит от текущего режима и подрежима.

        Args:
            event: Событие клика wxPython.
        """
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
            if self._handle_reselection(point_abs):
                return

            if self.submode == self.SUBMODE_SEGMENT:
                if self.selected_point:
                    p1_rel = self.abs_to_rel(self.selected_point)
                    p2_rel = point_rel
                    self.rel_lines.append((p1_rel, p2_rel, self.line_type))
                    self.selected_point = None
                    self.update_all_intersections()
                    if self.line_type != self.LINE_AUX:
                        self.update_endpoint_points()
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
            if is_border:
                pass
            else:
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
        """
        Попытаться завершить построение биссектрисы.

        Требует 3 точки: две точки угла и вершина угла.
        Затем ищет пересечение с выбранной линией.

        Returns:
            bool: True если биссектриса успешно построена.
        """
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
        inter = line_intersection_rel((v_rel, bis_end), (edge_p1_rel, edge_p2_rel))

        if inter:
            end_abs = self.rel_to_abs(inter)
            if end_abs:
                self.rel_lines.append((v_rel, inter, self.line_type))
                self.bisector_points = []
                self.update_all_intersections()
                self.update_endpoint_points()
                self.save_state()
                return True

        left, top, right, bottom = self.get_grid_bounds()
        bounds = [
            ((left, bottom), (right, bottom)),
            ((right, bottom), (right, top)),
            ((right, top), (left, top)),
            ((left, top), (left, bottom)),
        ]

        min_dist = float("inf")
        best_end = None

        for b1, b2 in bounds:
            inter = line_intersection_rel(
                (v_rel, bis_end), (self.abs_to_rel(b1), self.abs_to_rel(b2))
            )
            if inter:
                end_abs = self.rel_to_abs(inter)
                if end_abs:
                    dist = math.hypot(end_abs[0] - v[0], end_abs[1] - v[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_end = inter

        if best_end:
            self.rel_lines.append((v_rel, best_end, self.line_type))
            self.bisector_points = []
            self.update_all_intersections()
            self.update_endpoint_points()
            self.save_state()
            return True

        end_rel = (v_rel[0] + 500 * bis_dir[0], v_rel[1] + 500 * bis_dir[1])
        self.rel_lines.append((v_rel, end_rel, self.line_type))
        self.bisector_points = []
        self.update_all_intersections()
        self.update_endpoint_points()
        self.save_state()
        return True

    def try_finish_perpendicular(self, target_line):
        """
        Попытаться завершить построение перпендикуляра.

        Требует начальную точку и целевую линию.
        Опускает перпендикуляр из начальной точки на целевую линию.

        Args:
            target_line (tuple): Целевая линия ((x1, y1), (x2, y2)).

        Returns:
            bool: True если перпендикуляр успешно построен.
        """
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
        self.update_endpoint_points()
        self.save_state()
        return True

    def finish_incenter(self):
        """
        Завершить построение линий к инцентру треугольника.

        Требует 3 выбранные точки (вершины треугольника).
        Вычисляет инцентр и рисует линии от вершин к инцентру.
        """
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
        self.update_endpoint_points()
        self.save_state()

    def on_export(self, event):
        """
        Обработчик экспорта паттерна в файл .orp (JSON).

        Сохраняет все линии и параметры сетки в JSON файл.

        Args:
            event: Событие меню/кнопки wxPython.
        """

        with wx.FileDialog(
            self,
            "Экспорт в .orp",
            wildcard="Проект оригами (*.orp)|*.orp",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fileDialog.GetPath()
            if not pathname.lower().endswith(".orp"):
                pathname += ".orp"

            data = {"div_num": self.div_num, "lines": []}
            for p1_rel, p2_rel, ltype in self.rel_lines:
                data["lines"].append(
                    {
                        "p1": [round(p1_rel[0], 12), round(p1_rel[1], 12)],
                        "p2": [round(p2_rel[0], 12), round(p2_rel[1], 12)],
                        "type": ltype,
                    }
                )

            try:
                with open(pathname, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                wx.MessageBox(
                    f"Экспортировано: {os.path.basename(pathname)}",
                    "Успех",
                    wx.OK | wx.ICON_INFORMATION,
                )
            except Exception as e:
                wx.MessageBox(f"Ошибка экспорта:\n{e}", "Ошибка", wx.OK | wx.ICON_ERROR)

    def on_import(self, event):
        """
        Обработчик импорта паттерна из файла .orp (JSON).

        Загружает все линии и параметры сетки из JSON файла.

        Args:
            event: Событие меню/кнопки wxPython.
        """
        with wx.FileDialog(
            self,
            "Импорт .orp",
            wildcard="Проект оригами (*.orp)|*.orp",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fileDialog.GetPath()

            try:
                with open(pathname, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "div_num" not in data or "lines" not in data:
                    raise ValueError("Неверный формат .orp")

                self.div_num = int(data["div_num"])
                self.div_text.SetValue(str(self.div_num))
                self.update_grid_size()

                self.rel_lines = []
                for line in data["lines"]:
                    if line["type"] not in [
                        self.LINE_MOUNTAIN,
                        self.LINE_VALLEY,
                        self.LINE_AUX,
                    ]:
                        continue
                    p1_rel = tuple(line["p1"])
                    p2_rel = tuple(line["p2"])
                    self.rel_lines.append((p1_rel, p2_rel, line["type"]))

                self.update_all_intersections()
                self.update_endpoint_points()
                self.save_state()
                self.canvas.Refresh()

                wx.MessageBox(
                    f"Импортировано: {len(self.rel_lines)} линий",
                    "Успех",
                    wx.OK | wx.ICON_INFORMATION,
                )
            except Exception as e:
                wx.MessageBox(f"Ошибка импорта:\n{e}", "Ошибка", wx.OK | wx.ICON_ERROR)

    def on_check_foldability(self, event):
        """
        Обработчик кнопки "Проверить складываемость".

        Запускает полную проверку и показывает результат:
        - Если ошибок нет: показывает паттерн складок в отдельном окне
        - Если есть ошибки: показывает паттерн с выделенными красным проблемными вершинами

        Args:
            event: Событие кнопки wxPython.
        """
        result = self.check_foldability()
        if result["foldable"]:
            PatternViewer(self, result["crease_pattern"])
        else:
            error_vertices = [
                Point2D(p[0], p[1]) if isinstance(p, tuple) else p
                for p in result["problem_vertices"]
            ]
            ErrorViewer(self, result["crease_pattern"], error_vertices)

    def check_foldability(self) -> dict:
        """
        Выполнить полную проверку складываемости текущего паттерна.

        Проверяет условия Кавасаки для всех внутренних вершин.

        Returns:
            dict: Словарь с ключами:
                  - "foldable" (bool): Может ли паттерн быть сложен.
                  - "errors" (list): Список найденных ошибок.
                  - "problem_vertices" (list): Список проблемных вершин.
                  - "crease_pattern" (CreasePattern): Построенный паттерн.
        """
        cp = self.create_crease_pattern()
        cp.build_graph()

        errors = []
        problem_vertices = set()  # Набор проблемных вершин

        def is_border(p: Point2D, tol=1e-8) -> bool:
            """
            Проверить, находится ли точка на границе квадратного листа бумаги.

            Args:
                p (Point2D): Проверяемая точка с координатами x и y.
                tol (float): Допуск расстояния для определения граничной точки.

            Returns:
                bool: True, если точка находится на одной из четырёх граней квадрата.
                      False, если точка находится внутри квадрата или полностью вне его.
            """
            return (
                abs(p.x - 1.0) < tol
                or abs(p.x + 1.0) < tol
                or abs(p.y - 1.0) < tol
                or abs(p.y + 1.0) < tol
            )

        fold_vertices = defaultdict(list)

        for edge in cp.edges:
            if not edge.is_fold_edge():
                continue
            fold_vertices[edge.sv.p].append(edge)
            fold_vertices[edge.ev.p].append(edge)

        for v_point, incident_edges in fold_vertices.items():
            if not incident_edges:
                continue

            rays = []
            for edge in incident_edges:
                if edge.sv.p.is_close(v_point):
                    dir_vec = edge.ev.p - v_point
                else:
                    dir_vec = edge.sv.p - v_point
                angle = math.atan2(dir_vec.y, dir_vec.x)
                rays.append((edge.type, angle))

            rays.sort(key=lambda x: x[1])

            unique_rays = []
            for typ, ang in rays:
                if not unique_rays:
                    unique_rays.append((typ, ang))
                    continue
                prev_ang = unique_rays[-1][1]
                diff = (ang - prev_ang + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < 1e-8:
                    if unique_rays[-1][0] != typ:
                        errors.append(
                            f"Вершина {v_point}: конфликт типов складок в одном направлении"
                        )
                        problem_vertices.add(v_point)
                else:
                    unique_rays.append((typ, ang))

            n = len(unique_rays)
            if n == 0:
                continue

            is_border_vertex = is_border(v_point)

            if not is_border_vertex and n % 2 != 0:
                errors.append(
                    f"Вершина {v_point}: Нарушение первого правила Кавасаки - нечётное количество складок ({n})"
                )
                problem_vertices.add(v_point)

            if n < 2 and not is_border_vertex:
                errors.append(
                    f"Вершина {v_point}: слишком мало складок ({n}) для внутренней вершины"
                )
                problem_vertices.add(v_point)

            # Подсчёт M и V
            m_count = sum(1 for t, _ in unique_rays if t == LineType.MOUNTAIN)
            v_count = sum(1 for t, _ in unique_rays if t == LineType.VALLEY)

            if not is_border_vertex and n >= 2:
                if abs(m_count - v_count) != 2:
                    errors.append(
                        f"Вершина {v_point}: Нарушение второго правила Кавасаки - разность длин и гор не равна 2, |M−V|={abs(m_count - v_count)} (M={m_count}, V={v_count})"
                    )
                    problem_vertices.add(v_point)

            if n >= 3 and not is_border_vertex:
                angles = []
                for i in range(n):
                    a1 = unique_rays[i][1]
                    a2 = unique_rays[(i + 1) % n][1]
                    diff = (a2 - a1 + 2 * math.pi) % (2 * math.pi)
                    angles.append(diff)

                sum_even = sum(angles[i] for i in range(0, n, 2))
                sum_odd = sum(angles[i] for i in range(1, n, 2))

                if abs(math.degrees(sum_even) - 180.0) > 2.0:
                    errors.append(
                        "Нарушение третьего правила Кавасаки - сумма четных и нечетных углов не равна 180°"
                    )
                    errors.append(
                        f"Вершина {v_point}: сумма четных углов = {math.degrees(sum_even):.1f}° ≠ 180°"
                    )
                    problem_vertices.add(v_point)
                if abs(math.degrees(sum_odd) - 180.0) > 2.0:
                    errors.append(
                        f"Вершина {v_point}: Сумма нечётных углов = {math.degrees(sum_odd):.1f}° ≠ 180°"
                    )
                    problem_vertices.add(v_point)

        if errors:
            return {
                "foldable": False,
                "errors": errors,
                "problem_vertices": list(problem_vertices),
                "crease_pattern": cp,
            }

        return {
            "foldable": True,
            "errors": [],
            "problem_vertices": [],
            "crease_pattern": cp,
        }

    @staticmethod
    def _i(x):
        """
        Преобразовать координату в целое число для отрисовки.

        Args:
            x (float): Координата для преобразования.

        Returns:
            int: Округленное целое число.
        """
        return int(round(x))

    def on_paint(self, event):
        """
        Обработчик события рисования (перерисовка холста).

        Вызывает все методы отрисовки элементов паттерна в правильном порядке.

        Args:
            event: Событие рисования wxPython.
        """
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
        self.draw_black_points(dc)
        self.draw_hover_point(dc)
        self.draw_preview_line(dc)
        self.draw_coordinates(dc)

    def draw_grid_area(self, dc):
        """
        Нарисовать чёрную границу квадратной сетки.

        Args:
            dc: Контекст устройства wxPython.
        """
        left, top, right, bottom = self.get_grid_bounds()
        dc.SetPen(wx.Pen(wx.BLACK, 2))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(
            self._i(left), self._i(top), self._i(right - left), self._i(bottom - top)
        )

    def draw_grid_lines(self, dc):
        """
        Нарисовать серые линии сетки разделения.

        Args:
            dc: Контекст устройства wxPython.
        """
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
        """
        Нарисовать вспомогательные линии (AUX) серым цветом.

        Args:
            dc: Контекст устройства wxPython.
        """
        dc.SetPen(wx.Pen(wx.Colour(180, 180, 180), 1))
        for p1_abs, p2_abs, ltype in self.get_abs_lines():
            if ltype == self.LINE_AUX:
                dc.DrawLine(
                    self._i(p1_abs[0]),
                    self._i(p1_abs[1]),
                    self._i(p2_abs[0]),
                    self._i(p2_abs[1]),
                )

    def draw_main_lines(self, dc):
        """
        Нарисовать основные линии складок (красный для гор, синий для долин).

        Args:
            dc: Контекст устройства wxPython.
        """
        for p1_abs, p2_abs, ltype in self.get_abs_lines():
            if ltype == self.LINE_MOUNTAIN:
                dc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))
            elif ltype == self.LINE_VALLEY:
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))
            else:
                continue
            dc.DrawLine(
                self._i(p1_abs[0]),
                self._i(p1_abs[1]),
                self._i(p2_abs[0]),
                self._i(p2_abs[1]),
            )

    def draw_hovered_segment(self, dc):
        """
        Нарисовать выделенный зелёным цветом сегмент под курсором.

        Используется в режимах удаления и изменения типа.

        Args:
            dc: Контекст устройства wxPython.
        """
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
        """
        Нарисовать выбранные точки при построении биссектрисы.

        Args:
            dc: Контекст устройства wxPython.
        """
        if self.mode != self.MODE_INPUT or self.submode != self.SUBMODE_BISECTOR:
            return
        if not self.bisector_points:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180),
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        dc.SetBrush(wx.Brush(color))
        dc.SetPen(wx.Pen(color, 2))

        for p in self.bisector_points[:3]:
            x, y = p
            dc.DrawCircle(self._i(x), self._i(y), 6)

    def draw_incenter_points(self, dc):
        """
        Нарисовать выбранные вершины треугольника при построении инцентра.

        Args:
            dc: Контекст устройства wxPython.
        """
        if self.mode != self.MODE_INPUT or self.submode != self.SUBMODE_INCENTER:
            return
        if not self.incenter_points:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180),
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        dc.SetBrush(wx.Brush(color))
        dc.SetPen(wx.Pen(color, 2))

        for p in self.incenter_points:
            x, y = p
            dc.DrawCircle(self._i(x), self._i(y), 6)

    def draw_perp_point(self, dc):
        """
        Нарисовать начальную точку при построении перпендикуляра.

        Args:
            dc: Контекст устройства wxPython.
        """
        if self.mode != self.MODE_INPUT or self.submode != self.SUBMODE_PERP:
            return
        if not self.perp_start_point:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180),
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        dc.SetBrush(wx.Brush(color))
        dc.SetPen(wx.Pen(color, 2))
        x, y = self.perp_start_point
        dc.DrawCircle(self._i(x), self._i(y), 6)

    def draw_mode_preview(self, dc):
        """
        Нарисовать предпросмотр текущей операции (линии, биссектрисы и т.д.).

        Args:
            dc: Контекст устройства wxPython.
        """
        if self.mode != self.MODE_INPUT:
            return

        color_map = {
            self.LINE_MOUNTAIN: wx.Colour(255, 0, 0),
            self.LINE_VALLEY: wx.Colour(0, 0, 255),
            self.LINE_AUX: wx.Colour(180, 180, 180),
        }
        color = color_map.get(self.line_type, wx.Colour(255, 0, 0))

        if (
            self.submode == self.SUBMODE_SEGMENT
            and self.selected_point
            and self.hover_point
        ):
            dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
            sp = self.selected_point
            hp = self.hover_point
            dc.DrawLine(self._i(sp[0]), self._i(sp[1]), self._i(hp[0]), self._i(hp[1]))

        elif self.submode == self.SUBMODE_BISECTOR and self.bisector_points:
            dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
            pts = self.bisector_points[:3]
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                dc.DrawLine(
                    self._i(p1[0]), self._i(p1[1]), self._i(p2[0]), self._i(p2[1])
                )

            if len(pts) == 3:
                v = pts[1]
                dc.SetBrush(wx.Brush(color))
                dc.SetPen(wx.Pen(color, 2))
                dc.DrawCircle(self._i(v[0]), self._i(v[1]), 6)

            if self.bisector_waiting_for_edge and self.hovered_segment:
                idx, a, b, _, is_border = self.hovered_segment
                edge = self.square_edges[idx] if is_border else (a, b)

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
                            end_rel = (
                                v_rel[0] + 500 * bis_dir[0],
                                v_rel[1] + 500 * bis_dir[1],
                            )
                            end_abs = self.rel_to_abs(end_rel)
                            if end_abs:
                                dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
                                dc.DrawLine(
                                    self._i(v[0]),
                                    self._i(v[1]),
                                    self._i(end_abs[0]),
                                    self._i(end_abs[1]),
                                )

        elif self.submode == self.SUBMODE_INCENTER:
            for p in self.incenter_points:
                dc.SetBrush(wx.Brush(color))
                dc.SetPen(wx.Pen(color, 2))
                dc.DrawCircle(self._i(p[0]), self._i(p[1]), 6)

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
                            dc.DrawLine(
                                self._i(p[0]),
                                self._i(p[1]),
                                self._i(center_abs[0]),
                                self._i(center_abs[1]),
                            )

                        dc.SetBrush(wx.Brush(color))
                        dc.SetPen(wx.Pen(color, 2))
                        dc.DrawCircle(self._i(center_abs[0]), self._i(center_abs[1]), 6)

                else:
                    dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
                    for p in self.incenter_points:
                        dc.DrawLine(
                            self._i(p[0]),
                            self._i(p[1]),
                            self._i(self.hover_point[0]),
                            self._i(self.hover_point[1]),
                        )

        elif self.submode == self.SUBMODE_PERP:
            if self.perp_start_point and self.hovered_segment:
                start = self.perp_start_point
                try:
                    idx, a, b, _, is_border = self.hovered_segment
                except ValueError:
                    return
                p1, p2 = self.square_edges[idx] if is_border else (a, b)

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
                dc.DrawLine(
                    self._i(start[0]),
                    self._i(start[1]),
                    self._i(foot[0]),
                    self._i(foot[1]),
                )
                dc.SetBrush(wx.Brush(color))
                dc.SetPen(wx.Pen(color, 2))
                dc.DrawCircle(self._i(foot[0]), self._i(foot[1]), 6)

    def draw_black_points(self, dc):
        """
        Нарисовать чёрные точки для конечных точек и пересечений.

        Вспомогательные пересечения показываются серыми точками меньшего размера.

        Args:
            dc: Контекст устройства wxPython.
        """
        dc.SetBrush(wx.Brush(wx.BLACK))
        dc.SetPen(wx.Pen(wx.BLACK, 1))
        for rel_p in self.rel_endpoint_points:
            p = self.rel_to_abs(rel_p)
            if p:
                dc.DrawCircle(self._i(p[0]), self._i(p[1]), 3)
        for rel_p in self.rel_intersections:
            p = self.rel_to_abs(rel_p)
            if p:
                dc.DrawCircle(self._i(p[0]), self._i(p[1]), 3)
        for rel_p in self.rel_invisible:
            p = self.rel_to_abs(rel_p)
            if p:
                dc.SetBrush(wx.Brush(wx.Colour(100, 100, 100)))
                dc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 1))
                dc.DrawCircle(self._i(p[0]), self._i(p[1]), 2)

    def draw_hover_point(self, dc):
        """
        Нарисовать зелёную точку под курсором мыши.

        Args:
            dc: Контекст устройства wxPython.
        """
        if self.hover_point:
            x, y = self.hover_point
            dc.SetBrush(wx.Brush(wx.Colour(0, 255, 0)))
            dc.SetPen(wx.Pen(wx.Colour(0, 255, 0), 2))
            dc.DrawCircle(self._i(x), self._i(y), 5)

    def draw_preview_line(self, dc):
        """
        Нарисовать пунктирный предпросмотр линии при её рисовании.

        Показывает как будет выглядеть линия с начальной точки на текущую.

        Args:
            dc: Контекст устройства wxPython.
        """
        if (
            self.mode == self.MODE_INPUT
            and self.submode == self.SUBMODE_SEGMENT
            and self.selected_point
            and self.hover_point
        ):
            color = (
                wx.Colour(255, 0, 0)
                if self.line_type == self.LINE_MOUNTAIN
                else wx.Colour(0, 0, 255)
                if self.line_type == self.LINE_VALLEY
                else wx.Colour(180, 180, 180)
            )
            dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
            sp = self.selected_point
            hp = self.hover_point
            dc.DrawLine(self._i(sp[0]), self._i(sp[1]), self._i(hp[0]), self._i(hp[1]))

    def draw_coordinates(self, dc):
        """
        Нарисовать координаты текущей позиции мыши в относительных координатах.

        Args:
            dc: Контекст устройства wxPython.
        """
        if self.hover_point:
            rel = self.abs_to_rel(self.hover_point)
            if rel:
                x, y = rel
                text = f"({x:.3f}, {y:.3f})"
                dc.SetFont(
                    wx.Font(
                        10,
                        wx.FONTFAMILY_DEFAULT,
                        wx.FONTSTYLE_NORMAL,
                        wx.FONTWEIGHT_NORMAL,
                    )
                )
                dc.SetTextForeground(wx.Colour(100, 100, 100))
                tw, th = dc.GetTextExtent(text)
                pos = self.hover_point
                dc.DrawText(text, self._i(pos[0] - tw // 2), self._i(pos[1] - th - 10))

    def on_set_div(self, event):
        """
        Обработчик установки количества делений сетки.

        Проверяет что введено число от 1 до 100 и пересчитывает сетку.

        Args:
            event: Событие введения значения wxPython.
        """
        try:
            new_div = int(self.div_text.GetValue())
            if new_div < 1 or new_div > 100:
                raise ValueError
            self.div_num = new_div
            self.update_grid_size()
            self.save_state()
        except ValueError:
            wx.MessageBox(
                "Введите целое число от 1 до 100", "Ошибка", wx.OK | wx.ICON_ERROR
            )
            self.div_text.SetValue(str(self.div_num))


class PatternViewer(wx.Frame):
    """
    Окно для визуализации успешного паттерна складок (без ошибок).

    Показывает паттерн складок со всеми вершинами чёрными точками
    без выделения проблемных областей.

    Attributes:
        crease_pattern (CreasePattern): Паттерн для визуализации.
        panel (wx.Panel): Панель для рисования.
    """

    def __init__(self, parent, crease_pattern):
        """
        Инициализировать окно визуализации успешного паттерна.

        Args:
            parent: Родительское окно.
            crease_pattern (CreasePattern): Паттерн для отображения.
        """
        super().__init__(
            parent, title="Проверка складывания - Узор корректен", size=(800, 800)
        )
        self.crease_pattern = crease_pattern
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.Centre()
        self.Show()

    def on_paint(self, event):
        """
        Обработчик события рисования (перерисовка паттерна без ошибок).

        Рисует:
        - Границу квадрата чёрным цветом
        - Линии складок (красный для гор, синий для долин)
        - Все вершины чёрными точками

        Args:
            event: Событие рисования wxPython.
        """
        dc = wx.PaintDC(self.panel)
        dc.Clear()
        dc.SetBackground(wx.Brush(wx.Colour(255, 255, 255)))
        dc.Clear()

        w, h = self.panel.GetSize()
        margin = 40
        size = min(w, h) - 2 * margin
        scale = size / 2.0
        cx, cy = w // 2, h // 2

        # Рисуем границу квадрата
        dc.SetPen(wx.Pen(wx.BLACK, 2))
        dc.SetBrush(wx.Brush(wx.WHITE))
        dc.DrawRectangle(
            int(cx - scale), int(cy - scale), int(2 * scale), int(2 * scale)
        )

        for line in self.crease_pattern.lines:
            p0 = line.p0
            p1 = line.p1
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


if __name__ == "__main__":
    app = wx.App()
    frame = Foldify(None, "Foldify")
    app.MainLoop()
