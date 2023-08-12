from level.Constants import resolution
import itertools

import numpy as np
from shapely.geometry import Point


def get_contour_dims(contour):
    max_x, max_y = max(contour[:, 0]), max(contour[:, 1])
    min_x, min_y = min(contour[:, 0]), min(contour[:, 1])

    width = max_x - min_x
    height = max_y - min_y
    return dict(
        width = width.item(),
        height = height.item(),
        area = (width * height).item(),
        center_pos = np.asarray([min_x + width / 2, min_y + height / 2]),
        max_x = max_x,
        max_y = max_y,
        min_x = min_x,
        min_y = min_y
    )


def get_rectangles(contour, polygon):
    rectangles = []
    contour_list = list(contour)
    # Creates additional points on the contour to create functioning rectangles
    create_new_points(contour_list)

    enumerated_contours = list(enumerate(contour_list))

    rec_counter = 0
    for (idx_1, p1), (idx_2, p2), (idx_3, p3), (idx_4, p4) in itertools.combinations(enumerated_contours, 4):

        # Check diags
        diag_1 = np.abs(np.linalg.norm(p1 - p3))
        diag_2 = np.abs(np.linalg.norm(p2 - p4))
        if abs(diag_1 - diag_2) > 0.001:
            continue

        # Check if diagonal corners are orthogonal to each other
        if np.abs(np.dot(p1 - p2, p2 - p3)) > 0.01:
            continue

        if np.abs(np.dot(p3 - p4, p4 - p1)) > 0.01:
            continue

        # Check for every line if the center line is in a square
        does_intersect = False
        if polygon.convex_hull != polygon.area:
            for center_point in [(p1 + p2) / 2, (p2 + p3) / 2, (p3 + p4) / 2, (p1 + p4) / 2]:
                if not Point(center_point).intersects(polygon):
                    does_intersect = True
                    break

        if not does_intersect:
            rectangle = np.asarray([p1, p2, p3, p4]).reshape((4, 1, 2))
            rectangles.append(rectangle)

    return rectangles, contour_list


def get_rectangles_through_diags(contour):
    rectangles = []
    contour = contour.reshape(len(contour), 2)
    contour_list = list(contour)
    # Creates additional points on the contour to create functioning rectangles
    create_new_points(contour_list)

    enumerated_contours = list(enumerate(contour_list))

    counter = 0
    for (p1_idx_1, p1_1), (p1_idx_2, p1_2) in itertools.combinations(enumerated_contours, 2):

        # Check that it is not the next or the previous point in the line
        if p1_idx_1 + 1 == p1_idx_2 or p1_idx_1 - 1 == p1_idx_2:
            continue

        # Check that the point of a diagonal are not horizontal
        if np.min(np.abs(p1_1 - p1_2)) <= 0.01:
            continue

        for (p2_idx_1, p2_1), (p2_idx_2, p2_2) in itertools.combinations(enumerated_contours, 2):

            # Check that it is not the next or the previous point in the line
            if p2_idx_1 + 1 == p2_idx_2 or p2_idx_1 - 1 == p2_idx_2:
                continue

            # Check if the two diagonals share a point
            if p1_idx_1 == p2_idx_1 or p1_idx_1 == p2_idx_2 or p1_idx_2 == p2_idx_1 or p1_idx_2 == p2_idx_2:
                continue

            if np.min(np.abs(p1_1 - p1_2)) <= 0.01:
                continue

            diag_1 = np.abs(np.linalg.norm(p1_1 - p1_2))
            diag_2 = np.abs(np.linalg.norm(p2_1 - p2_2))

            if abs(diag_1 - diag_2) < 0.0001:
                rectangle = np.asarray([p1_1, p2_1, p1_2, p2_2]) \
                    .reshape((4, 1, 2))
                rectangles.append(rectangle)

        counter += 1

    return rectangles, contour_list


def get_next_line(contour_list, idx):
    if idx + 1 >= len(contour_list):
        return (idx, contour_list[-1]), (0, contour_list[0])

    return (idx, contour_list[idx]), (idx + 1, contour_list[idx + 1])


def create_new_points(contour_list: list):

    # Search for inner corners
    counter_1 = 0
    while counter_1 < len(contour_list):
        (idx_1, p1), (idx_2, p2) = get_next_line(contour_list, idx = counter_1)
        (idx_3, p3), (idx_4, p4) = get_next_line(contour_list, idx = counter_1 + 2)

        # Check if it is a straight line
        if np.min(np.abs(p2 - p3)) >= 0.001:
            if np.dot(p1 - p2, p2 - p3) > 0.01:
                cord = intersecting_line(p1, p2, p3, p4)
                if cord is not False:
                    contour_list.insert(idx_2 + 1, np.asarray(cord))
        counter_1 += 1

    added_points = []

    global_closest = 1000
    points = np.asarray(contour_list)
    for point_idx, point in enumerate(points):
        global_closest = np.min([np.min(np.linalg.norm(np.delete(points, point_idx, axis = 0) - point, axis = 1)), global_closest])

    counter_1 = 0
    while counter_1 < len(contour_list):
        (p1_idx_1, p1_1), (p1_idx_2, p1_2) = get_next_line(contour_list, idx = counter_1)
        # Check if it is a straight line
        if np.min(np.abs(p1_1 - p1_2)) >= 0.001:
            counter_1 += 1
            continue

        counter_2 = 0

        while counter_2 + counter_1 + 2 < len(contour_list):
            (p2_idx_1, p2_1), (p2_idx_2, p2_2) = get_next_line(contour_list, idx = counter_2 + counter_1 + 2)

            # if p1_idx_1 == 0 and p1_idx_2 == 1 and p2_idx_1 == 4 and p2_idx_2 == 5:
            #     pass

            if p1_idx_1 == p2_idx_1 or p1_idx_1 == p2_idx_2 or p1_idx_2 == p2_idx_1 or p1_idx_2 == p2_idx_2:
                counter_2 += 1
                continue

            if np.min(np.abs(p2_2 - p2_1)) >= 0.001:
                counter_2 += 1
                continue

            cord = intersecting_line(p1_1, p1_2, p2_1, p2_2)
            if not cord:
                counter_2 += 1
                continue

            d1 = np.linalg.norm(p1_1 - cord)
            d2 = np.linalg.norm(p1_2 - cord)
            d3 = np.linalg.norm(p2_1 - cord)
            d4 = np.linalg.norm(p2_2 - cord)
            if d1 < 0.01 or d2 < 0.01 or d3 < 0.01 or d4 < 0.01:
                counter_2 += 1
                continue

            isnt_on_first_line = d1 + d2 - np.linalg.norm(p1_2 - p1_1) > 0.01
            isnt_on_second_line = d3 + d4 - np.linalg.norm(p2_2 - p2_1) > 0.01
            if isnt_on_first_line and isnt_on_second_line:
                counter_2 += 1
                continue

            cord_array = np.asarray(cord)
            # Check if the point exists allready

            min_distance = np.inf

            if len(added_points) > 0:
                min_distance = np.min(np.linalg.norm(np.asarray(added_points) - cord_array, axis = 1))
                if min_distance < 0.01:
                    counter_1 += 1
                    continue

            if global_closest > min_distance:
                global_closest = min_distance

            # If its on the first line then insert it between the first and second
            if not isnt_on_first_line:

                direct_vec = (p1_2 - p1_1) * (1 / np.linalg.norm(p1_2 - p1_1)) * global_closest

                add_point(cord_array - direct_vec, contour_list + added_points, added_points)
                added_points.append(cord_array)
                add_point(cord_array + direct_vec, contour_list + added_points, added_points)

            else:
                # Otherwise between the second two points
                direct_vec = (p2_2 - p2_1) * (1 / np.linalg.norm(p2_2 - p2_1)) * global_closest

                add_point(cord_array - direct_vec, contour_list + added_points, added_points)
                added_points.append(cord_array)
                add_point(cord_array + direct_vec, contour_list + added_points, added_points)

            counter_2 += 2

        counter_1 += 1

    added_points = np.asarray(added_points)

    contour_list_copy = contour_list.copy()
    if len(added_points) != 0:
        counter_1 = 0
        # Insert dots into contour ring at right positions
        added_points_counter = 0
        to_be_added_points = len(added_points)
        while counter_1 < len(contour_list_copy):
            (idx_1, p1), (idx_2, p2) = get_next_line(contour_list_copy, idx = counter_1)

            d_1 = np.linalg.norm(added_points - p1, axis = 1)
            d_2 = np.linalg.norm(added_points - p2, axis = 1)
            select_mask = d_1 + d_2 - np.linalg.norm(p2 - p1) < 0.01

            points_on_line = added_points[select_mask]

            for new_point_idx, point_idx in enumerate(np.argsort(d_1[select_mask])):
                contour_list.insert(idx_1 + new_point_idx + 1 + added_points_counter, points_on_line[point_idx])

            added_points_counter += len(points_on_line)

            added_points = added_points[~select_mask]
            counter_1 += 1
            if added_points_counter >= to_be_added_points:
                break

def add_point(new_point, previous_points, point_list):
    min_distance = np.min(np.linalg.norm(np.asarray(previous_points) - new_point, axis = 1))
    if min_distance > 0.01:
        point_list.append(new_point)
        return True
    return False


# https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
def chunker(seq, size):
    return (seq[pos:pos + size] if pos + size < len(seq) else [seq[-1], seq[0]] for pos in range(0, len(seq), 1))


# Intersecting lines https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def intersecting_line(p1, p2, p3, p4):
    l1 = line(p1, p2)
    l2 = line(p3, p4)

    return intersection(l1, l2)

# https://stackoverflow.com/questions/44505504/how-to-make-a-circular-kernel

def get_circular_kernel(diameter):

    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(np.uint8)

    return kernel