from ._pattern_data_model import PatternDataModel
from ._ang_binned_statistic import ang_binned_statistic
import pyqtgraph as pg

class AngularStatisticViewer(pg.PlotWidget):
    def __init__(self, dm: PatternDataModel, bins=10, parent=None):
        super().__init__(parent=parent)
        self.dataLine = pg.PlotDataItem()
        self.addItem(self.dataLine)
        self.bins = bins

    def updatePlot(self, pv):
        dm = pv.currentDataset
        ans = ang_binned_statistic(
            dm.patterns[dm.index],
            dm.detector,
            bins=self.bins
        )
        self.dataLine.setData(
            (ans.bin_edges[:-1] + ans.bin_edges[1:])/2,
            ans.statistic
        )
