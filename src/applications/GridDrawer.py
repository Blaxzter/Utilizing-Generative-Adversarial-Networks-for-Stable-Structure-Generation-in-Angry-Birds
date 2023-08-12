from tkinter import *
from tkinter.ttk import Notebook
from typing import Dict, List

import numpy as np

from applications import TkinterUtils
from level import Constants
from util import Utils
from util.tkinterutils.ScrollableNotebook import ScrollableNotebook


class GridDrawer:
    def __init__(self, level_drawer, frame, draw_mode, draw_area_width, draw_area_height, level_height, level_width):
        self.level_drawer = level_drawer
        self.left_frame = frame

        self.draw_mode = draw_mode

        self.draw_area_width = draw_area_width
        self.draw_area_height = draw_area_height

        self.default_level_height = level_height
        self.default_level_width = level_width

        self.colors = ['#a8d399', '#aacdf6', '#f98387', '#dbcc81']

        self.tab_control = ScrollableNotebook(
            self.left_frame,
            wheelscroll = True, tabmenu = True,
            tab_change_callback = lambda event:
                setattr(self, 'selected_tab', self.tab_control.notebookTab.index(self.tab_control.notebookTab.select()))
        )
        self.tabs: List[Dict] = []
        self.selected_tab = 0
        self.create_tab_panes(2)

        self.material_id = 1

        self.init_blocks()

    def clear_tab_panes(self):
        for tab_idx, tab in enumerate(self.tabs):
            for key, element in tab.items():
                invert_op = getattr(element, "destroy", None)
                if callable(invert_op):
                    invert_op()

        for item in self.tab_control.winfo_children():
            item.destroy()

        self.tabs = []

    def create_tab_panes(self, panel_amounts = 1):
        self.tabs = self.tab_control.clear_tab_panes(self.tabs)

        for tab_index in range(panel_amounts):
            create_tab = Frame(self.tab_control)
            self.tab_control.add(create_tab, text = 'Main Tab' if tab_index == 0 else f'Layer {tab_index}')
            self.tabs.append(dict(
                frame = create_tab,
            ))

        self.tab_control.pack(expand = 1, fill = "both")

        for tab_index in range(panel_amounts):
            self.create_draw_canvas(tab = tab_index)
            self.create_grid(tab = tab_index)

    def create_draw_canvas(self, tab = -1):
        if tab == -1: tab = self.selected_tab
        draw_canvas = Canvas(self.frame(tab), width = self.draw_area_width, height = self.draw_area_height)
        draw_canvas.pack(expand = YES, side = LEFT, pady = (30, 30), padx = (30, 30))

        draw_canvas.bind("<B1-Motion>", self.paint)
        draw_canvas.bind("<Button-1>", self.paint)
        draw_canvas.bind("<Motion>", self.hover)
        draw_canvas.bind("<B3-Motion>", lambda event: self.paint(event, clear = True))
        draw_canvas.bind("<Button-3>", lambda event: self.paint(event, clear = True))
        self.tabs[tab]['canvas'] = draw_canvas
        self.tabs[tab]['level_height'] = self.default_level_height
        self.tabs[tab]['level_width'] = self.default_level_width

    def set_level_dims(self, height, width, tab = -1):
        if tab == -1: tab = self.selected_tab

        self.tabs[tab]['level_height'] = height
        self.tabs[tab]['level_width'] = width

    def delete_grids(self):
        for tab in self.tabs:
            tab['canvas'].delete('grid_line')

    def create_grids(self):
        for tab_index in range(len(self.tabs)):
            self.create_grid(tab_index)

    def create_grid(self, tab = -1):
        if tab == -1: tab = self.selected_tab
        level_width, level_height = self.get_level_dims(tab)
        self.tabs[tab]['canvas'].delete('grid_line')
        elements = np.zeros((level_height + 1, level_width + 1))
        rectangles = [[None for _ in range(elements.shape[0])] for _ in range(elements.shape[1])]

        self.rec_height = self.draw_area_height / level_height
        self.rec_width = self.draw_area_width / level_width
        self.draw_grid_lines(tab)

        self.tabs[tab]['elements'] = elements
        self.tabs[tab]['rectangles'] = rectangles

    def draw_grid_lines(self, tab = -1):
        if tab == -1: tab = self.selected_tab
        level_width, level_height = self.get_level_dims(tab)
        for i in range(level_height):
            self.canvas(tab).create_line(
                self.rec_height * i, self.rec_height,
                self.rec_height * i, self.draw_area_height - self.rec_height,
                width = 1,
                tag = 'grid_line'
            )

        for i in range(level_width):
            self.canvas(tab).create_line(
                self.rec_width, self.rec_width * i,
                self.draw_area_width - self.rec_width, self.rec_width * i,
                width = 1,
                tag = 'grid_line'
            )

    def paint(self, event, clear = False, tab = -1):
        if tab == -1: tab = self.selected_tab
        self.hover(event)

        fill_color = self.colors[self.material_id - 1]

        x_index = round((event.x + (self.rec_width / 2) - 0) / self.rec_width)
        y_index = round((event.y + (self.rec_height / 2) - 0) / self.rec_height)

        block_name, x_axis, x_pos, y_axis, y_pos, rotated = self.get_block_position(x_index, y_index, self.selected_block)

        if not clear:
            self.set_block(block_name, fill_color, self.material_id, x_axis, x_pos, y_axis, y_pos, rotated = rotated)

        single_element = self.draw_mode.get() != '' and self.draw_mode.get() != 'LevelImg'
        level_width, level_height = self.get_level_dims(tab)
        if clear:
            for x, x_idx in zip(x_pos, x_axis):
                for y, y_idx in zip(y_pos, y_axis):
                    if y_idx > 1 and y_idx < level_height and x_idx > 1 and x_idx < level_width:

                        if block_name == 'bird':
                            x_center = np.average(x_pos)
                            y_center = np.average(y_pos)
                            r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) / np.max(
                                [x_pos - x_center, y_pos - y_center])
                            if r > 1.2:
                                continue

                        # Get element of current tab
                        draw_canvas = self.canvas(tab)
                        elements = self.elements(tab)
                        rects = self.rects(tab)

                        elements[y_idx, x_idx] = 0
                        if single_element:
                            draw_canvas.delete(str((x_idx, y_idx)))
                        else:
                            draw_canvas.delete(rects[y_idx][x_idx])
                            rects[y_idx][x_idx] = None

    def hover(self, event, tab = -1):
        if tab == -1: tab = self.selected_tab
        draw_canvas = self.canvas(tab)
        draw_canvas.delete('hover')

        fill_color = self.colors[self.material_id - 1]
        lighten_fill_color = Utils.lighten_color(fill_color, 0.5)

        x_index = round((event.x + (self.rec_width / 2) - 0) / self.rec_width)
        y_index = round((event.y + (self.rec_height / 2) - 0) / self.rec_height)

        block_name, x_axis, x_pos, y_axis, y_pos, rotated = self.get_block_position(x_index, y_index, self.selected_block)

        single_element = self.draw_mode.get() != 'LevelImg'
        level_width, level_height = self.get_level_dims(tab)

        for x_pos_idx, (x, x_idx) in enumerate(zip(x_pos, x_axis)):
            for y_pos_idx, (y, y_idx) in enumerate(zip(y_pos, y_axis)):
                if 1 < y_idx < level_height and 1 < x_idx < level_width:

                    if block_name == 'bird':
                        x_center = np.average(x_pos)
                        y_center = np.average(y_pos)
                        r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) / np.max(
                            [x_pos - x_center, y_pos - y_center])
                        if r > 1.2:
                            continue

                    x1, y1 = int(x - self.rec_width / 2), int(y - self.rec_width / 2)
                    x2, y2 = int(x + self.rec_height / 2), int(y + self.rec_height / 2)

                    next_color = fill_color
                    if single_element:
                        if x_pos_idx == int(len(x_axis) / 2) and y_pos_idx == int(len(y_axis) / 2):
                            next_color = lighten_fill_color

                    draw_canvas.create_rectangle(x1, y1, x2, y2, fill = next_color, tag = f'hover')

    def get_block_position(self, x_index, y_index, block):
        block_width, block_height, block_name = block
        x_axis = np.array([x for x in range(x_index, x_index + block_width)])
        y_axis = np.array([y for y in range(y_index, y_index + block_height)])
        x_pos = x_axis * self.rec_width - self.rec_width / 2
        y_pos = y_axis * self.rec_height - self.rec_height / 2
        return block_name, x_axis, x_pos, y_axis, y_pos, block_height > block_width

    def set_block(self, block_name, fill_color, material_id, x_axis, x_pos, y_axis, y_pos, tab = -1, rotated = False):
        if tab == -1: tab = self.selected_tab

        single_element = self.draw_mode.get() != '' and self.draw_mode.get() != 'LevelImg'
        lighten_fill_color = Utils.lighten_color(fill_color, 0.6)

        centerpos = (x_axis[int(len(x_axis) / 2)], y_axis[int(len(y_axis) / 2)])
        level_width, level_height = self.get_level_dims(tab)

        for x, x_idx in zip(x_pos, x_axis):
            for y, y_idx in zip(y_pos, y_axis):
                x1, y1 = int(x - self.rec_width / 2), int(y - self.rec_width / 2)
                x2, y2 = int(x + self.rec_height / 2), int(y + self.rec_height / 2)

                if 1 < y_idx < level_height and 1 < x_idx < level_width:

                    if block_name == 'bird':
                        x_center = np.average(x_pos)
                        y_center = np.average(y_pos)
                        r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) / np.max(
                            [x_pos - x_center, y_pos - y_center])
                        if r > 1.2:
                            continue

                    # Only original color the middle element
                    next_color = fill_color
                    if single_element:
                        if x_idx == centerpos[0] and y_idx == centerpos[1]:
                            next_color = lighten_fill_color

                    # Get element of current tab
                    draw_canvas = self.canvas(tab)
                    elements = self.elements(tab)
                    rects = self.rects(tab)

                    if elements[y_idx][x_idx] == 0:
                        next_tag = ['rectangle'] + ([str(centerpos)] if single_element else [])
                        rects[y_idx][x_idx] = \
                            draw_canvas.create_rectangle(
                                x1, y1, x2, y2, fill = next_color,
                                tag = next_tag,
                                outline = fill_color)

                        # Only set center element
                        if single_element:
                            if x_idx == centerpos[0] and y_idx == centerpos[1]:
                                if block_name == 'bird':
                                    elements[y_idx, x_idx] = 40
                                else:
                                    elements[y_idx, x_idx] = (list(Constants.block_names.values()).index(
                                        block_name) + 1 + (1 if rotated else 0)) + (material_id - 1) * 13
                        else:
                            elements[y_idx, x_idx] = material_id

    def block_cursor(self, block):
        self.selected_block = (block['width'], block['height'], block['name'])
        if block['name'] == 'bird':
            self.material_id = 4

        for button in self.level_drawer.selected_button.values():
            button.config(bg = 'grey')

        self.level_drawer.selected_button[block['button']].config(bg = 'lightblue')

    def delete_drawing(self, tab = -1):
        if tab == -1: tab = self.selected_tab

        level_width, level_height = self.get_level_dims(tab)

        # Get element of current tab
        draw_canvas = self.canvas(tab)
        elements = self.elements(tab)
        draw_canvas.delete('rectangle')

        rects = [[None for _ in range(elements.shape[0])] for _ in range(elements.shape[1])]
        elements = np.zeros((level_height + 1, level_width + 1))

        self.tabs[tab]['rectangles'] = rects
        self.tabs[tab]['elements'] = elements

    def init_blocks(self):

        blocks_placed = 0

        prev_blocks = []

        level_width, level_height = self.get_level_dims(0)

        while blocks_placed <= 4:

            random_x = np.random.randint(3, level_width - 4)
            random_y = np.random.randint(3, level_height - 4)

            block_amount = len(self.level_drawer.block_data)
            random_block = self.level_drawer.block_data[str(np.random.randint(block_amount))]
            material = np.random.randint(1, 4)

            new_block = (random_block['width'] + 1, random_block['height'] + 1, random_block['name'])

            block_name, x_axis, x_pos, y_axis, y_pos, rotated = self.get_block_position(random_x, random_y, new_block)
            xx, yy = np.meshgrid(x_axis, y_axis)
            positions = np.vstack([xx.ravel(), yy.ravel()])

            if np.max(x_axis) > level_width or np.max(y_axis) > level_height:
                continue

            overlapping = False
            for comp_pos in prev_blocks:
                pos_stack = np.hstack([positions, comp_pos])
                if np.unique(pos_stack, axis = 1).shape[-1] != pos_stack.shape[-1]:
                    overlapping = True
                    break

            if overlapping:
                continue

            self.set_block(block_name, self.colors[material - 1], material, x_axis, x_pos, y_axis, y_pos)
            blocks_placed += 1
            prev_blocks.append(positions)

    def get_level_dims(self, i):
        return self.get_level_width(i), self.get_level_height(i)

    def get_level_height(self, i):
        return self.tabs[i]['level_height']

    def get_level_width(self, i):
        return self.tabs[i]['level_width']

    def frame(self, i):
        return self.tabs[i]['frame']

    def current_canvas(self):
        return self.tabs[self.selected_tab]['canvas']

    def canvas(self, i):
        return self.tabs[i]['canvas']

    def rects(self, i):
        return self.tabs[i]['rectangles']

    def current_elements(self):
        return self.tabs[self.selected_tab]['elements']

    def elements(self, i):
        return self.tabs[i]['elements']
