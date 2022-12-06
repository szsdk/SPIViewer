import numpy as np
import pyqtgraph as pg
import emcfile as ef
from pyqtgraph.Qt import QtWidgets
from spi_viewer.pattern_viewer import PatternDataModel, PatternViewer

app = QtWidgets.QApplication([])
pd = PatternDataModel(
    ef.PatternsSOneEMC("/u/szsdk/NeoEMC/data/photons.emc"),
    detector=ef.detector("/u/szsdk/NeoEMC/data/det_sim.dat"),
)
w = PatternViewer(pd)

w.show()

# btn = QtWidgets.QPushButton("ff")
# btn.clicked.connect(lambda: pd.updateSelectedList(np.random.choice(pd.pattern2DDet.num_data, size=20)))
# btn.show()

pg.exec()
