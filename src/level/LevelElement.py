from __future__ import annotations

from typing import Optional

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, Point

from level import Constants
from level.Constants import ObjectType
from util.Utils import round_cord


class LevelElement:

    def __init__(self, id, type, material = None, x = None, y = None, rotation = 0, scaleX = 1, scaleY = 1):
        self.id = id
        self.type = type

        if x is None:
            self.object_type = ObjectType.Bird
            return

        self.material = material
        self.x = round(float(x) * Constants.coordinate_round) / Constants.coordinate_round
        self.y = round(float(y) * Constants.coordinate_round) / Constants.coordinate_round

        self.original_x = self.x
        self.original_y = self.y

        if rotation is not None:
            self.rotation = float(rotation)
            self.is_vertical = False
            if round(self.rotation / 90) in [1, 3]:
                self.is_vertical = True

        self.coordinates = np.array([self.x, self.y])

        self.object_type: Optional[ObjectType] = None

        if self.type == "Platform":
            self.size = round_cord(float(scaleX) * 0.62,  float(scaleY) * 0.62)
            self.object_type = ObjectType.Platform

        elif self.type in Constants.pig_types.values():
            self.size = Constants.pig_size
            self.object_type = ObjectType.Pig
            # if self.y <= -3.5 + Constants.pig_size[0] / 2:
            #     self.y = -3.5 + Constants.pig_size[0] / 2

        elif self.type in Constants.additional_objects.values():
            self.index = list(Constants.additional_objects.values()).index(self.type) + 1
            self.size = Constants.additional_object_sizes[str(self.index)]
            self.object_type = ObjectType.SpecialBlock

        elif self.type == "Slingshot":
            self.size = (0.25, 0.75)
            self.object_type = ObjectType.Slingshot
        else:
            self.index = list(Constants.block_names.values()).index(self.type) + 1
            if self.is_vertical:
                self.index += 1
            self.size = Constants.block_sizes[self.index]
            self.object_type = ObjectType.Block

        self.width = self.size[0]
        self.height = self.size[1]

        self.int_width = round(self.width / Constants.resolution)
        self.int_height = round(self.height / Constants.resolution)

        self.shape_polygon: Optional[Polygon] = None

        # self.required_dots = round(((self.size[0] - Constants.resolution) / Constants.resolution) * ((self.size[1] - Constants.resolution) / Constants.resolution))
        # self.used_dots = 0

    def get_bottom_left(self):
        horizontal_offset = self.size[0] / 2
        vertical_offset = self.size[1] / 2
        return round_cord(self.x - horizontal_offset, self.y - vertical_offset)

    def create_set_geometry(self):
        self.shape_polygon = self.create_geometry()

    def create_geometry(self):

        horizontal_offset = self.size[0] / 2
        vertical_offset = self.size[1] / 2

        if self.type in Constants.pig_types.values():
            return Point(self.x, self.y).buffer(self.size[0] / 2)
        elif self.type in Constants.additional_objects.values():
            if "Circle" in self.type:
                return Point(self.x, self.y).buffer(self.size[0] / 2)
            if "Triangle" in self.type:
                p1 = round_cord(self.x - horizontal_offset, self.y - vertical_offset)
                p2 = round_cord(self.x + horizontal_offset, self.y - vertical_offset)
                p3 = round_cord(self.x + horizontal_offset, self.y + vertical_offset)

                poly = Polygon([p1, p2, p3])
                return affinity.rotate(poly, self.rotation - 90, 'center')
        else:
            p1 = round_cord(self.x + horizontal_offset, self.y + vertical_offset)
            p2 = round_cord(self.x + horizontal_offset, self.y - vertical_offset)
            p3 = round_cord(self.x - horizontal_offset, self.y - vertical_offset)
            p4 = round_cord(self.x - horizontal_offset, self.y + vertical_offset)

            return Polygon([p1, p2, p3, p4])

    def distance(self, o: LevelElement):
        if not self.shape_polygon.disjoint(o.shape_polygon):
            return 0

        return self.shape_polygon.distance(o.shape_polygon)

    def get_identifier(self):
        if "Basic" in self.type:
            return 4
        if "Platform" in self.type:
            return 5

        return Constants.materials.index(self.material) + 1

    def __str__(self):
        return f"id: {self.id} " \
               f"type: {self.type} " \
               f"material: {self.material} " \
               f"x: {self.x} " \
               f"y: {self.y} " \
               f"rotation: {self.rotation} " \
               f"size: {self.size} "


