from loguru import logger
from shapely import affinity
from shapely.geometry import Polygon, Point

ref_point = Point(0, 0)

p1 = (1, 0)
p2 = (2, 0)
p3 = (2, 1)

poly = Polygon([p1, p2, p3])
rot_poly = affinity.rotate(poly, 180, 'center')

logger.debug(ref_point.distance(rot_poly))
