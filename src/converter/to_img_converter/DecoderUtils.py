import numpy as np


def recalibrate_blocks(created_level_elements, flipped = False):
    if len(created_level_elements) == 0:
        return []

    # Post process
    sorted_elements = sorted(created_level_elements, key = lambda _element: (_element.y, _element.x))
    for element_idx, element in enumerate(sorted_elements):
        # search for element lower then itself
        for lower_element in sorted_elements[:element_idx]:
            if element.shape_polygon.overlaps(lower_element.shape_polygon):

                if abs(lower_element.y - element.y) <= 0.1:
                    # move element upwards out of the overlapping element
                    left_x = lower_element.x + lower_element.width / 2
                    right_x = element.x - element.width / 2

                    move_upwards = abs(left_x - right_x)
                    element.x += move_upwards
                else:
                    # move element upwards out of the overlapping element
                    lower_element_y = lower_element.y + lower_element.height / 2
                    upper_element_y = element.y - element.height / 2

                    move_upwards = abs(upper_element_y - lower_element_y)
                    element.y += move_upwards

    # move everything to zero
    lowest_value = np.min(list(map(lambda _element: _element.y - element.height / 2, sorted_elements)))
    for element_idx, element in enumerate(sorted_elements):
        element.y -= lowest_value

    return sorted_elements


def get_img_trim(level_img):

    left_space = 0
    for row_idx in range(level_img.shape[1]):
        if len(np.nonzero(level_img[:, row_idx])[0]) == 0:
            left_space += 1
        else:
            break

    right_space = 0
    for row_idx in range(1, level_img.shape[1]):
        if len(np.nonzero(level_img[:, level_img.shape[1] - row_idx])[0]) == 0:
            right_space += 1
        else:
            break

    top_space = 0
    for cloumn_idx in range(level_img.shape[0]):
        if len(np.nonzero(level_img[cloumn_idx, :])[0]) == 0:
            top_space += 1
        else:
            break

    bottom_space = 0
    for cloumn_idx in range(1, level_img.shape[0]):
        if len(np.nonzero(level_img[level_img.shape[0] - cloumn_idx, :])[0]) == 0:
            bottom_space += 1
        else:
            break

    return top_space, bottom_space, left_space, right_space

def trim_img(level_img, ret_trims = False):
    top_space, bottom_space, left_space, right_space = get_img_trim(level_img)

    bottom_trim = level_img.shape[0] - bottom_space
    right_trim = level_img.shape[1] - right_space
    if ret_trims:
        return level_img[top_space: bottom_trim, left_space: right_trim], (top_space, bottom_space, left_space, right_space)
    else:
        return level_img[top_space: bottom_trim, left_space: right_trim]
