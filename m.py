import pyqtgraph as pg
import emcfile as ef
import neoemc as ne
from pyqtgraph.Qt import QtWidgets
from spi_viewer.patter_viewer import PatternDataModel, PatternViewer

app = QtWidgets.QApplication([])
pd = PatternDataModel(
    ne.photon_dataset.pattern_2ddet(
        ef.PatternsSOneEMC("/Users/sz/NeoEMC/data/photons.emc"),
        ef.detector("/Users/sz/NeoEMC/data/det_sim.dat")
    )
)
w = PatternViewer(pd)
w.show()
pg.exec()
