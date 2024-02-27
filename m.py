import logging

import emcfile as ef
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from rich.logging import RichHandler

from spiviewer import ImageDataModel, PatternDataModel, patternViewer
from spiviewer.plugins import ROIExporterDialog

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
    return ef.detector(
        coor=coor,
        factor=factor,
        mask=mask,
        detd=detd,
        ewald_rad=detd,
        check_consistency=False,
    )


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
        "images": ImageDataModel(np.random.rand(10, 5, 10), None),
        "empty": PatternDataModel(
            patterns, detector=det, initIndex=None, selectedList=np.arange(0)
        ),
    }
)
# w = patternViewer("/u/szsdk/NeoEMC/data/photons.emc", "/u/szsdk/NeoEMC/data/det_sim.dat")
p = pg.EllipseROI([0, 0], [10, 10], pen="r")
w.imageViewer.view.addItem(p)

w.currentImageChangedFunc = lambda x: x.infoLabel.update(
    {"doubled index": 2 * x.datasetsManager.dataset1.rawIndex}
)


w.imageViewer.scene.exportDialog = ROIExporterDialog(w.imageViewer.scene)
w.show()

pg.exec()
