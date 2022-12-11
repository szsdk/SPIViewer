import logging
from rich.logging import RichHandler

import numpy as np
import pyqtgraph as pg
import emcfile as ef
from pyqtgraph.Qt import QtWidgets
from spi_viewer.pattern_viewer import PatternDataModel, PatternViewer, patternViewer

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(message)s", handlers=[RichHandler()]
)


def fake_detector(s, detd, beamstop):
    xy = (
        np.array(np.meshgrid(*(np.linspace(-13, 13, s, endpoint=True),) * 2))
        .reshape(2, -1)
        .T
    )
    r = np.linalg.norm(xy, axis=1)
    z = detd - np.sqrt(detd**2 - r**2)
    coor = np.empty((xy.shape[0], 3))
    coor[:, :2] = xy
    coor[:, 2] = z
    factor = np.ones(xy.shape[0])
    mask = np.zeros_like(factor, int)
    mask[r < beamstop] = 2
    return ef.detector(coor=coor, factor=factor, mask=mask, detd=detd, ewald_rad=detd)


det = fake_detector(64, 150, 4)
num_data = 211
patterns = (5 * np.random.rand(num_data, det.num_pix) ** 4).astype(int)
patterns[:, det.mask == 2] = 0
patterns = ef.patterns(patterns)

app = QtWidgets.QApplication([])
# w = patternViewer(
#     "/Users/sz/NeoEMC/data/photons.emc", detector="/Users/sz/NeoEMC/data/det_sim.dat"
# )
pd = PatternDataModel(patterns, detector=det, modify=False)
w = patternViewer(
    {
        "(default)": pd,
        "a": PatternDataModel(patterns, detector=det, selectedList=np.arange(11)),
    }
)

w.show()

# btn = QtWidgets.QPushButton("ff")
# btn.clicked.connect(
#     # lambda: w.removeDataset("a")
#     lambda: w.setDataset("b", PatternDataModel(
#             patterns,
#             detector=det,
#             selectedList=np.arange(11, 30)
#         ))
# )
# btn.show()

pg.exec()
