import matplotlib.pyplot as plt
import matplotlib


class Container:
    def __init__(self, identifier, posx, posy, color=None, alpha=0.4, height=0.1, width=0.1) -> None:
        self.id = identifier
        self.color = color
        self.alpha = alpha
        self.height = height
        self.width = width
        self.posx = posx
        self.posy = posy

    def plot(self, ax):
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (self.posx, self.posy),
                self.width,
                self.height,
                color=self.color,
                alpha=self.alpha)
        )
        plt.text(x=self.posx + (self.width / 2), y=self.posy + (self.height / 2), s=self.id,
                 color='black', ha='center', va='center', fontsize=12)
