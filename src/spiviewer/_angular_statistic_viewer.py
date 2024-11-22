import pyqtgraph as pg

from ._ang_binned_statistic import ang_binned_statistic
from ._pattern_data_model import PatternDataModel


class AngularStatisticViewer(pg.PlotWidget):
    def __init__(self, dm: PatternDataModel, bins=10, parent=None):
        super().__init__(parent=parent)
        self.dataLine = pg.PlotDataItem()
        self.addItem(self.dataLine)
        self.bins = bins
        self.setLabel("bottom", "q/pixel")

    def updatePlot(self, pv):
        dm = pv.datasetsManager.dataset1
        ans = ang_binned_statistic(
            dm.patterns[dm.rawIndex], dm.detector, bins=self.bins
        )
        self.dataLine.setData(
            (ans.bin_edges[:-1] + ans.bin_edges[1:]) / 2, ans.statistic
        )
