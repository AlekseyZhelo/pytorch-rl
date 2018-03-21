from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Circle(xy=center, radius=height / 2.0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerRectangle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.Rectangle(xy=(0.5 * width - 0.5 * xdescent - 0.5 * height, -ydescent), width=height,
                               height=height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
