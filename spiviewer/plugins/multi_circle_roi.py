import pyqtgraph as pg


class MultiCircleROI(pg.ROI):
    def __init__(self, **args):
        self.circles = []
        pg.ROI.__init__(self, [0, 0], size=[1, 1], **args)

    def getState(self):
        state = pg.ROI.getState(self)
        state.update({f"p{i:08d}": p.getState() for i, p in enumerate(self.circles)})
        return state

    def saveState(self):
        state = pg.ROI.saveState(self)
        state.update({f"p{i:08d}": p.saveState() for i, p in enumerate(self.circles)})
        return state

    def setState(self, state):
        pg.ROI.setState(state)
        for p, (_, s) in zip(self.circles, sorted(state.items())):
            p.setState(s)

    def paint(self, p, *args):
        pass

    def clearCircles(self):
        for p in self.circles:
            self.scene().removeItem(p)
        self.circles = []

    def addCircle(self, pos, r, **args):
        circle = pg.CircleROI(pos, r, parent=self, **args)
        for handler in circle.handles:
            handler["item"].hide()
        self.circles.append(circle)
        return circle

    def updatePeaks(self, peaks, peakColors):
        self.clearCircles()
        if len(peaks) == 0:
            return
        ncolor = max(0, peakColors.max() + 1)
        # for c, color in zip(peaks - self.peakDiameter / 2, peakColors):
        for c, color in zip(peaks - 2, peakColors):
            circle = self.addCircle(
                c,
                2,
                pen="w" if color == -1 else pg.intColor(color, hues=ncolor),
                movable=False,
                rotatable=False,
                resizable=False,
            )
            for handler in circle.handles:
                handler["item"].hide()
