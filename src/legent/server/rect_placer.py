from typing import Tuple
from pyqtree import Index

#二维平面放置矩形保证不会相互重叠
#用到了self spindex(spatial index空间索引)
class RectPlacer:
    def __init__(self, bbox: Tuple[float, float, float, float]) -> None:
        """
        Args:
            bbox (Tuple[float, float, float, float]): (xmin, ymin, xmax, ymax)
        """
        self.bbox = bbox#元组，定义了这块场地的总边界，所有矩形都放在这个边界之内
        self.spindex = Index(bbox=bbox)#创建核心的空间索引对象，告诉它总的活动范围是bbox

    def place_rectangle(#核心放置函数，在一个指定的位置bbox放置一个名为name的矩形，进行重叠检查
        self, name: str, bbox: Tuple[float, float, float, float]
    ) -> bool:
        """place a rectangle into the 2d space without overlapping

        Args:
            name (str): rectangle name
            bbox (Tuple[float, float, float, float]): (xmin, ymin, xmax, ymax)

        Returns:
            bool: whether successfully placed without overlapping
        """
        matches = self.spindex.intersect(bbox)
        if matches:
            return False
        else:
            self.spindex.insert(name, bbox)
            return True

    def place(self, name, x, z, x_size, z_size):
        """place a rectangle into the 2d space without overlapping

        Args:
            name (str): rectangle name
            x (float): x position
            z (float): z position
            x_size (float): x size
            z_size (float): z size

        Returns:
            bool: whether successfully placed without overlapping
        """
        return self.place_rectangle(
            name, (x - x_size / 2, z - z_size / 2, x + x_size / 2, z + z_size / 2)
        )

    def insert(self, name: str, bbox: Tuple[float, float, float, float]):
        """force place a rectangle into the 2d space"""
        self.spindex.insert(name, bbox)
