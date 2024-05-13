import matplotlib.pyplot as plt
from .container import Container


class Edge:
    def __init__(self, src: Container, dst: Container, width=1) -> None:
        self.src = src
        self.dst = dst
        self.color = src.color
        self.width = width
        self.src_x = src.posx + src.width
        self.src_y = src.posy + src.height / 2
        self.dst_x = dst.posx
        self.dst_y = dst.posy + dst.height / 2

    def plot(self, ax):
        ax.plot(
            [self.src_x, self.dst_x],
            [self.src_y, self.dst_y],
            color=self.color,
            linewidth=self.width,
        )
