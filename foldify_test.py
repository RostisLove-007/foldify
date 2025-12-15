import pytest
import math
from foldify import (
    Point2D, LineType, LayerRelation, OriLine, OriVertex, OriEdge,
    line_intersection_rel, point_to_line_distance,
    reflect_point_over_line, triangle_incenter,
    _merge_close_points, EPS
)

class TestPoint2D:
    """Тесты класса Point2D."""

    def test_point_creation(self):
        """Положительный тест: создание точки."""
        p = Point2D(3.0, 4.0)
        assert p.x == 3.0
        assert p.y == 4.0

    def test_point_creation_zero(self):
        """Положительный тест: создание нулевой точки."""
        p = Point2D(0, 0)
        assert p.x == 0
        assert p.y == 0

    def test_point_creation_negative(self):
        """Положительный тест: создание точки с отрицательными координатами."""
        p = Point2D(-5.5, -2.3)
        assert p.x == -5.5
        assert p.y == -2.3

    def test_point_addition(self):
        """Положительный тест: сложение двух точек."""
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)
        result = p1 + p2
        assert result.x == 4
        assert result.y == 6

    def test_point_addition_with_negatives(self):
        """Положительный тест: сложение с отрицательными значениями."""
        p1 = Point2D(5, -3)
        p2 = Point2D(-2, 7)
        result = p1 + p2
        assert result.x == 3
        assert result.y == 4

    def test_point_subtraction(self):
        """Положительный тест: вычитание двух точек."""
        p1 = Point2D(5, 7)
        p2 = Point2D(2, 3)
        result = p1 - p2
        assert result.x == 3
        assert result.y == 4

    def test_point_subtraction_negative_result(self):
        """Положительный тест: вычитание с отрицательным результатом."""
        p1 = Point2D(2, 3)
        p2 = Point2D(5, 7)
        result = p1 - p2
        assert result.x == -3
        assert result.y == -4

    def test_point_multiplication_by_scalar(self):
        """Положительный тест: умножение точки на скаляр."""
        p = Point2D(2, 3)
        result = p * 3
        assert result.x == 6
        assert result.y == 9

    def test_point_multiplication_by_zero(self):
        """Положительный тест: умножение на ноль."""
        p = Point2D(5, 7)
        result = p * 0
        assert result.x == 0
        assert result.y == 0

    def test_point_multiplication_by_negative(self):
        """Положительный тест: умножение на отрицательное число."""
        p = Point2D(2, -3)
        result = p * (-2)
        assert result.x == -4
        assert result.y == 6

    def test_point_division_by_scalar(self):
        """Положительный тест: деление точки на скаляр."""
        p = Point2D(6, 9)
        result = p / 3
        assert result.x == 2
        assert result.y == 3

    def test_point_division_by_negative(self):
        """Положительный тест: деление на отрицательное число."""
        p = Point2D(4, -8)
        result = p / (-2)
        assert result.x == -2
        assert result.y == 4

    def test_point_division_by_very_small_number(self):
        """Положительный тест: деление на очень маленькое число."""
        p = Point2D(1, 1)
        result = p / 0.001
        assert result.x == 1000
        assert result.y == 1000

    def test_point_length_pythagorean_triple(self):
        """Положительный тест: длина вектора (пифагорова тройка 3-4-5)."""
        p = Point2D(3, 4)
        assert p.length() == 5.0

    def test_point_length_zero(self):
        """Положительный тест: длина нулевого вектора."""
        p = Point2D(0, 0)
        assert p.length() == 0

    def test_point_length_negative_coords(self):
        """Положительный тест: длина вектора с отрицательными координатами."""
        p = Point2D(-3, -4)
        assert p.length() == 5.0

    def test_point_distance_to_same_point(self):
        """Положительный тест: расстояние до одной и той же точки."""
        p1 = Point2D(5, 5)
        p2 = Point2D(5, 5)
        assert p1.distance_to(p2) == 0

    def test_point_distance_to_different_point(self):
        """Положительный тест: расстояние между разными точками."""
        p1 = Point2D(0, 0)
        p2 = Point2D(3, 4)
        assert p1.distance_to(p2) == 5.0

    def test_point_distance_negative_coordinates(self):
        """Положительный тест: расстояние с отрицательными координатами."""
        p1 = Point2D(-1, -1)
        p2 = Point2D(2, 3)
        expected = math.sqrt(9 + 16)  # sqrt(25) = 5
        assert abs(p1.distance_to(p2) - 5.0) < 1e-10

    def test_point_normalized_unit_vector(self):
        """Положительный тест: нормализация вектора длины > 1."""
        p = Point2D(3, 4)
        normalized = p.normalized()
        assert abs(normalized.length() - 1.0) < 1e-10

    def test_point_normalized_already_unit(self):
        """Положительный тест: нормализация уже единичного вектора."""
        p = Point2D(1, 0)
        normalized = p.normalized()
        assert abs(normalized.x - 1.0) < 1e-10
        assert abs(normalized.y - 0) < 1e-10

    def test_point_normalized_zero_vector(self):
        """Положительный тест: нормализация нулевого вектора."""
        p = Point2D(0, 0)
        normalized = p.normalized()
        assert normalized.x == 0
        assert normalized.y == 0

    def test_point_normalized_very_small_vector(self):
        """Положительный тест: нормализация очень маленького вектора."""
        p = Point2D(1e-12, 1e-12)
        normalized = p.normalized()
        # Из-за проверки на 1e-10, это считается нулевым вектором
        assert normalized.x == 0
        assert normalized.y == 0

    def test_point_dot_perpendicular_vectors(self):
        """Положительный тест: скалярное произведение перпендикулярных векторов."""
        p1 = Point2D(1, 0)
        p2 = Point2D(0, 1)
        assert p1.dot(p2) == 0

    def test_point_dot_parallel_vectors(self):
        """Положительный тест: скалярное произведение параллельных векторов."""
        p1 = Point2D(2, 0)
        p2 = Point2D(3, 0)
        assert p1.dot(p2) == 6

    def test_point_dot_opposite_vectors(self):
        """Положительный тест: скалярное произведение противоположных векторов."""
        p1 = Point2D(2, 3)
        p2 = Point2D(-2, -3)
        assert p1.dot(p2) == -13

    def test_point_cross_perpendicular_vectors(self):
        """Положительный тест: векторное произведение перпендикулярных векторов."""
        p1 = Point2D(1, 0)
        p2 = Point2D(0, 1)
        assert p1.cross(p2) == 1

    def test_point_cross_parallel_vectors(self):
        """Положительный тест: векторное произведение параллельных векторов."""
        p1 = Point2D(2, 3)
        p2 = Point2D(4, 6)
        assert p1.cross(p2) == 0

    def test_point_cross_antiparallel_vectors(self):
        """Положительный тест: векторное произведение антипараллельных векторов."""
        p1 = Point2D(1, 0)
        p2 = Point2D(0, -1)
        assert p1.cross(p2) == -1

    def test_point_perpendicular_vector(self):
        """Положительный тест: перпендикулярный вектор."""
        p = Point2D(3, 4)
        perp = p.perpendicular()
        assert perp.x == -4
        assert perp.y == 3
        # Проверяем, что они действительно перпендикулярны
        assert abs(p.dot(perp)) < 1e-10

    def test_point_perpendicular_twice(self):
        """Положительный тест: два раза перпендикулярный вектор = инверсия."""
        p = Point2D(2, 3)
        double_perp = p.perpendicular().perpendicular()
        assert abs(double_perp.x + p.x) < 1e-10
        assert abs(double_perp.y + p.y) < 1e-10

    def test_point_rotated_90_degrees(self):
        """Положительный тест: поворот на 90 градусов."""
        p = Point2D(1, 0)
        rotated = p.rotated(math.pi / 2)
        assert abs(rotated.x - 0) < 1e-10
        assert abs(rotated.y - 1) < 1e-10

    def test_point_rotated_180_degrees(self):
        """Положительный тест: поворот на 180 градусов."""
        p = Point2D(1, 0)
        rotated = p.rotated(math.pi)
        assert abs(rotated.x + 1) < 1e-10
        assert abs(rotated.y - 0) < 1e-10

    def test_point_rotated_360_degrees(self):
        """Положительный тест: поворот на 360 градусов = исходный вектор."""
        p = Point2D(3, 4)
        rotated = p.rotated(2 * math.pi)
        assert abs(rotated.x - p.x) < 1e-10
        assert abs(rotated.y - p.y) < 1e-10

    def test_point_rotated_45_degrees(self):
        """Положительный тест: поворот на 45 градусов."""
        p = Point2D(1, 0)
        rotated = p.rotated(math.pi / 4)
        expected_x = math.cos(math.pi / 4)
        expected_y = math.sin(math.pi / 4)
        assert abs(rotated.x - expected_x) < 1e-10
        assert abs(rotated.y - expected_y) < 1e-10

    def test_point_is_close_identical_points(self):
        """Положительный тест: идентичные точки близки."""
        p1 = Point2D(1, 2)
        p2 = Point2D(1, 2)
        assert p1.is_close(p2)

    def test_point_is_close_within_tolerance(self):
        """Положительный тест: точки в пределах допуска близки."""
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.0 + 1e-11, 2.0 + 1e-11)
        assert p1.is_close(p2)

    def test_point_is_close_beyond_tolerance(self):
        """Отрицательный тест: точки вне допуска не близки."""
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.0 + 1e-9, 2.0 + 1e-9)
        assert not p1.is_close(p2)

    def test_point_is_close_custom_tolerance(self):
        """Положительный тест: использование пользовательского допуска."""
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.1, 2.1)
        assert p1.is_close(p2, tol=0.2)

    def test_point_repr(self):
        """Положительный тест: строковое представление точки."""
        p = Point2D(1.5, 2.5)
        repr_str = repr(p)
        assert "1.5" in repr_str or "1.500000" in repr_str
        assert "2.5" in repr_str or "2.500000" in repr_str

class TestLineType:
    """Тесты перечисления LineType."""

    def test_line_type_mountain(self):
        """Положительный тест: тип MOUNTAIN."""
        assert LineType.MOUNTAIN.value == 1

    def test_line_type_valley(self):
        """Положительный тест: тип VALLEY."""
        assert LineType.VALLEY.value == 2

    def test_line_type_aux(self):
        """Положительный тест: тип AUX."""
        assert LineType.AUX.value == 3

    def test_line_type_cut(self):
        """Положительный тест: тип CUT."""
        assert LineType.CUT.value == 4

    def test_line_type_none(self):
        """Положительный тест: тип NONE."""
        assert LineType.NONE.value == 0

    def test_line_type_equality(self):
        """Положительный тест: сравнение типов линий."""
        assert LineType.MOUNTAIN == LineType.MOUNTAIN
        assert LineType.MOUNTAIN != LineType.VALLEY

class TestLayerRelation:
    """Тесты перечисления LayerRelation."""

    def test_layer_relation_above(self):
        """Положительный тест: слой выше."""
        assert LayerRelation.ABOVE == 1

    def test_layer_relation_below(self):
        """Положительный тест: слой ниже."""
        assert LayerRelation.BELOW == -1

    def test_layer_relation_unknown(self):
        """Положительный тест: неопределённое соотношение."""
        assert LayerRelation.UNKNOWN == 0

class TestOriLine:
    """Тесты класса OriLine."""

    def test_oriline_creation(self):
        """Положительный тест: создание линии."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        assert line.p0 == p0
        assert line.p1 == p1
        assert line.type == LineType.MOUNTAIN

    def test_oriline_default_type(self):
        """Положительный тест: тип по умолчанию AUX."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1)
        assert line.type == LineType.AUX

    def test_oriline_is_mountain(self):
        """Положительный тест: проверка типа MOUNTAIN."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        assert line.is_mountain()

    def test_oriline_is_not_mountain(self):
        """Отрицательный тест: не MOUNTAIN."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.VALLEY)
        assert not line.is_mountain()

    def test_oriline_is_valley(self):
        """Положительный тест: проверка типа VALLEY."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.VALLEY)
        assert line.is_valley()

    def test_oriline_is_not_valley(self):
        """Отрицательный тест: не VALLEY."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        assert not line.is_valley()

    def test_oriline_is_fold_line_mountain(self):
        """Положительный тест: складка для MOUNTAIN."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        assert line.is_fold_line()

    def test_oriline_is_fold_line_valley(self):
        """Положительный тест: складка для VALLEY."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.VALLEY)
        assert line.is_fold_line()

    def test_oriline_is_not_fold_line_aux(self):
        """Отрицательный тест: не складка для AUX."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.AUX)
        assert not line.is_fold_line()

    def test_oriline_zero_length_raises_error(self):
        """Отрицательный тест: линия нулевой длины вызывает ошибку."""
        p = Point2D(1, 1)
        with pytest.raises(ValueError):
            OriLine(p, p, LineType.MOUNTAIN)

class TestOriVertex:
    """Тесты класса OriVertex."""

    def test_orivertex_creation(self):
        """Положительный тест: создание вершины."""
        p = Point2D(1, 2)
        v = OriVertex(p)
        assert v.p == p

    def test_orivertex_degree_empty(self):
        """Положительный тест: степень пустой вершины."""
        p = Point2D(1, 2)
        v = OriVertex(p)
        assert v.degree() == 0

    def test_orivertex_hash(self):
        """Положительный тест: хешируемость вершины."""
        p = Point2D(1, 2)
        v1 = OriVertex(p)
        v2 = OriVertex(p)
        # Разные объекты должны иметь одинаковый хеш если точки одинаковые
        assert hash(v1) == hash(v2)

    def test_orivertex_equality_same_coords(self):
        """Положительный тест: равенство вершин с одинаковыми координатами."""
        p = Point2D(1, 2)
        v1 = OriVertex(p)
        v2 = OriVertex(p)
        assert v1 == v2

    def test_orivertex_inequality_different_coords(self):
        """Отрицательный тест: неравенство вершин с разными координатами."""
        v1 = OriVertex(Point2D(1, 2))
        v2 = OriVertex(Point2D(3, 4))
        assert v1 != v2

    def test_orivertex_equality_close_coords(self):
        """Положительный тест: равенство вершин с близкими координатами."""
        v1 = OriVertex(Point2D(1.0, 2.0))
        v2 = OriVertex(Point2D(1.0 + 1e-11, 2.0 + 1e-11))
        assert v1 == v2

class TestOriEdge:
    """Тесты класса OriEdge."""

    def test_oriedge_creation(self):
        """Положительный тест: создание ребра."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        sv = OriVertex(p0)
        ev = OriVertex(p1)
        edge = OriEdge(sv, ev, line)
        assert edge.sv == sv
        assert edge.ev == ev
        assert edge.type == LineType.MOUNTAIN

    def test_oriedge_is_fold_edge_mountain(self):
        """Положительный тест: ребро складки MOUNTAIN."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        sv = OriVertex(p0)
        ev = OriVertex(p1)
        edge = OriEdge(sv, ev, line)
        assert edge.is_fold_edge()

    def test_oriedge_is_fold_edge_valley(self):
        """Положительный тест: ребро складки VALLEY."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.VALLEY)
        sv = OriVertex(p0)
        ev = OriVertex(p1)
        edge = OriEdge(sv, ev, line)
        assert edge.is_fold_edge()

    def test_oriedge_is_not_fold_edge_aux(self):
        """Отрицательный тест: ребро не складки AUX."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.AUX)
        sv = OriVertex(p0)
        ev = OriVertex(p1)
        edge = OriEdge(sv, ev, line)
        assert not edge.is_fold_edge()

    def test_oriedge_equality(self):
        """Положительный тест: равенство рёбер (по идентификатору)."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        sv = OriVertex(p0)
        ev = OriVertex(p1)
        edge1 = OriEdge(sv, ev, line)
        edge2 = OriEdge(sv, ev, line)
        assert edge1 != edge2

    def test_oriedge_hashable(self):
        """Положительный тест: рёбра хешируемы."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)
        sv = OriVertex(p0)
        ev = OriVertex(p1)
        edge = OriEdge(sv, ev, line)
        edges_set = {edge}
        assert edge in edges_set

class TestLineIntersection:
    """Тесты функции line_intersection_rel."""

    def test_perpendicular_lines_intersection(self):
        """Положительный тест: пересечение перпендикулярных линий."""
        line1 = ((0, 0), (2, 0))  # Горизонтальная линия
        line2 = ((1, -1), (1, 1))  # Вертикальная линия
        result = line_intersection_rel(line1, line2)
        assert result is not None
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 0.0) < 1e-10

    def test_parallel_lines_no_intersection(self):
        """Отрицательный тест: параллельные линии не пересекаются."""
        line1 = ((0, 0), (2, 0))
        line2 = ((0, 1), (2, 1))
        result = line_intersection_rel(line1, line2)
        assert result is None

    def test_colinear_lines_no_intersection(self):
        """Отрицательный тест: коллинеарные линии не пересекаются."""
        line1 = ((0, 0), (2, 0))
        line2 = ((1, 0), (3, 0))
        result = line_intersection_rel(line1, line2)
        assert result is None

    def test_crossing_lines_intersection(self):
        """Положительный тест: пересечение диагональных линий."""
        line1 = ((0, 0), (2, 2))
        line2 = ((0, 2), (2, 0))
        result = line_intersection_rel(line1, line2)
        assert result is not None
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10

    def test_intersection_at_endpoint(self):
        """Положительный тест: пересечение в конечной точке."""
        line1 = ((0, 0), (2, 0))
        line2 = ((2, 0), (2, 2))
        result = line_intersection_rel(line1, line2)
        assert result is not None
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[1] - 0.0) < 1e-10

    def test_non_intersecting_segments(self):
        """Отрицательный тест: отрезки не пересекаются (близко, но вне отрезков)."""
        line1 = ((0, 0), (1, 1))
        line2 = ((2, 0), (3, 1))
        result = line_intersection_rel(line1, line2)
        assert result is None

class TestPointToLineDistance:
    """Тесты функции point_to_line_distance."""

    def test_point_on_line(self):
        """Положительный тест: точка на линии."""
        p = (1, 0)
        a = (0, 0)
        b = (2, 0)
        distance = point_to_line_distance(p, a, b)
        assert abs(distance) < 1e-10

    def test_point_perpendicular_to_line(self):
        """Положительный тест: точка над линией."""
        p = (1, 1)
        a = (0, 0)
        b = (2, 0)
        distance = point_to_line_distance(p, a, b)
        assert abs(distance - 1.0) < 1e-10

    def test_point_before_line_segment(self):
        """Положительный тест: точка перед началом отрезка."""
        p = (-1, 1)
        a = (0, 0)
        b = (2, 0)
        distance = point_to_line_distance(p, a, b)
        expected = math.hypot(-1, 1)
        assert abs(distance - expected) < 1e-10

    def test_point_after_line_segment(self):
        """Положительный тест: точка после конца отрезка."""
        p = (3, 1)
        a = (0, 0)
        b = (2, 0)
        distance = point_to_line_distance(p, a, b)
        expected = math.hypot(1, 1)
        assert abs(distance - expected) < 1e-10

    def test_point_to_zero_length_line(self):
        """Положительный тест: расстояние до линии нулевой длины."""
        p = (1, 1)
        a = (0, 0)
        b = (0, 0)
        distance = point_to_line_distance(p, a, b)
        expected = math.hypot(1, 1)
        assert abs(distance - expected) < 1e-10

class TestReflectPointOverLine:
    """Тесты функции reflect_point_over_line."""

    def test_reflect_point_over_horizontal_line(self):
        """Положительный тест: отражение над горизонтальной линией."""
        p = (1, 1)
        a = (0, 0)
        b = (2, 0)
        reflected = reflect_point_over_line(p, a, b)
        assert abs(reflected[0] - 1.0) < 1e-10
        assert abs(reflected[1] + 1.0) < 1e-10

    def test_reflect_point_over_vertical_line(self):
        """Положительный тест: отражение над вертикальной линией."""
        p = (1, 1)
        a = (0, 0)
        b = (0, 2)
        reflected = reflect_point_over_line(p, a, b)
        assert abs(reflected[0] + 1.0) < 1e-10
        assert abs(reflected[1] - 1.0) < 1e-10

    def test_reflect_point_on_line(self):
        """Положительный тест: отражение точки на линии."""
        p = (1, 0)
        a = (0, 0)
        b = (2, 0)
        reflected = reflect_point_over_line(p, a, b)
        assert abs(reflected[0] - p[0]) < 1e-10
        assert abs(reflected[1] - p[1]) < 1e-10

    def test_reflect_point_over_zero_length_line(self):
        """Положительный тест: отражение над линией нулевой длины."""
        p = (1, 1)
        a = (0, 0)
        b = (0, 0)
        reflected = reflect_point_over_line(p, a, b)
        assert reflected == p

class TestTriangleIncenter:
    """Тесты функции triangle_incenter."""

    def test_incenter_equilateral_triangle(self):
        """Положительный тест: инцентр равностороннего треугольника."""
        a = (0, 0)
        b = (2, 0)
        c = (1, math.sqrt(3))
        center = triangle_incenter(a, b, c)
        expected_x = (a[0] + b[0] + c[0]) / 3
        expected_y = (a[1] + b[1] + c[1]) / 3
        assert abs(center[0] - expected_x) < 1e-8
        assert abs(center[1] - expected_y) < 1e-8

    def test_incenter_right_triangle(self):
        """Положительный тест: инцентр прямоугольного треугольника."""
        a = (0, 0)
        b = (3, 0)
        c = (0, 4)
        center = triangle_incenter(a, b, c)
        assert 0 < center[0] < 3
        assert 0 < center[1] < 4

    def test_incenter_degenerate_triangle(self):
        """Положительный тест: инцентр вырожденного треугольника."""
        a = (0, 0)
        b = (0, 0)
        c = (1, 1)
        center = triangle_incenter(a, b, c)
        assert center == a

class TestMergeClosePoints:
    """Тесты функции _merge_close_points."""

    def test_merge_empty_list(self):
        """Положительный тест: объединение пустого списка."""
        result = _merge_close_points([])
        assert result == []

    def test_merge_single_point(self):
        """Положительный тест: один точки."""
        points = [Point2D(1, 1)]
        result = _merge_close_points(points)
        assert len(result) == 1

    def test_merge_distinct_points(self):
        """Положительный тест: не объединяемые точки."""
        points = [Point2D(1, 1), Point2D(3, 3), Point2D(5, 5)]
        result = _merge_close_points(points)
        assert len(result) == 3

    def test_merge_close_points(self):
        """Положительный тест: объединение близких точек."""
        points = [Point2D(1, 1), Point2D(1 + 1e-11, 1 + 1e-11)]
        result = _merge_close_points(points)
        assert len(result) == 1

    def test_merge_multiple_close_points(self):
        """Положительный тест: объединение нескольких близких точек."""
        points = [
            Point2D(0, 0),
            Point2D(1, 1),
            Point2D(1 + 1e-11, 1 + 1e-11),
            Point2D(2, 2),
            Point2D(2 + 1e-11, 2 + 1e-11)
        ]
        result = _merge_close_points(points)
        assert len(result) == 3

class TestIntegration:
    """Интеграционные тесты."""

    def test_create_and_validate_line(self):
        """Положительный тест: создание и валидация линии."""
        p0 = Point2D(0, 0)
        p1 = Point2D(1, 1)
        line = OriLine(p0, p1, LineType.MOUNTAIN)

        assert line.is_mountain()
        assert line.is_fold_line()
        assert line.length() == math.sqrt(2)

    def test_vector_chain_operations(self):
        """Положительный тест: цепь операций с векторами."""
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)
        p3 = Point2D(5, 6)

        v1 = p2 - p1
        v2 = p3 - p2

        assert v1.cross(v2) == 0

        combined = v1 + v2
        assert combined.length() == 4 * math.sqrt(2)

    def test_point_transformations_roundtrip(self):
        """Положительный тест: трансформации точек туда-обратно."""
        p = Point2D(1, 0)

        rotated = p.rotated(math.pi / 4)
        back = rotated.rotated(-math.pi / 4)

        assert p.is_close(back)