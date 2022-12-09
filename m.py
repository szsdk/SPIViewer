import numpy as np
import pyqtgraph as pg
import emcfile as ef
from pyqtgraph.Qt import QtWidgets
from spi_viewer.pattern_viewer import PatternDataModel, PatternViewer


def fake_detector(s, detd, beemstop):
    xy = (
        np.mgrid[-(s - 1) / 2.0 : (s + 1) / 2.0, -(s - 1) / 2.0 : (s + 1) / 2.0]
        .reshape(2, -1)
        .T
    )
    r = np.linalg.norm(xy, axis=1)
    z = detd - np.sqrt(detd**2 - r**2)
    coor = np.empty((xy.shape[0], 3))
    coor[:, :2] = xy
    coor[:, 2] = z
    factor = np.ones(xy.shape[0])
    mask = (r < beemstop).astype(int)
    return ef.detector(coor=coor, factor=factor, mask=mask, detd=detd, ewald_rad=detd)


det = fake_detector(64, 150, 4)
num_data = 200
patterns = (5 * np.random.rand(num_data, det.num_pix) ** 4).astype(int)
patterns[:, det.mask == 1] = 0
patterns = ef.patterns(patterns)

app = QtWidgets.QApplication([])
pd = PatternDataModel(
    patterns,
    detector=det,
    modify=False
)
w = PatternViewer(
    {
        "(default)": pd,
        "a": PatternDataModel(
            patterns,
            detector=det,
            selectedList=np.arange(10)
        )
    }
)

w.show()

# btn = QtWidgets.QPushButton("ff")
# btn.clicked.connect(lambda: pd.updateSelectedList(np.random.choice(pd.pattern2DDet.num_data, size=20)))
# btn.show()

pg.exec()
