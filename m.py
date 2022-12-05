import numpy as np
import pyqtgraph as pg
import emcfile as ef
import neoemc as ne
from pyqtgraph.Qt import QtWidgets
from spi_viewer.pattern_viewer import PatternDataModel, PatternViewer

app = QtWidgets.QApplication([])
pd = PatternDataModel(
    ne.photon_dataset.pattern_2ddet(
        ef.PatternsSOneEMC("/Users/sz/NeoEMC/data/photons.emc"),
        ef.detector("/Users/sz/NeoEMC/data/det_sim.dat")
    )
)
w = PatternViewer(pd)
w.show()

btn = QtWidgets.QPushButton("ff")
btn.clicked.connect(lambda: pd.updateSelectedList(np.random.choice(pd.pattern2DDet.num_data, size=20)))
btn.show()

pg.exec()
