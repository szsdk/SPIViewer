import operator
import numpy as np
import cachetools
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt

# pg.setConfigOptions(antialias=True)

__all__ = [
    "PatternDataModel",
    "PatternViewer",
]

class PatternDataModel(QtCore.QObject):
    selected = QtCore.pyqtSignal(int, np.ndarray)
    def __init__(self, patterns2DDet, initIndex=0, selectedList=None):
        super().__init__()
        self.pattern2DDet = patterns2DDet
        self.selectedList = np.arange(self.pattern2DDet.num_data) if selectedList is None else selectedList
        self._cache = cachetools.LRUCache(maxsize=32)
        self._idx = initIndex

    def select(self, idx: int):
        self._idx = idx
        self.selected.emit(*self.getSelection())

    def getSelection(self):
        rawIndex = int(self.selectedList[self._idx])
        return rawIndex, self._getImage(rawIndex)

    @cachetools.cachedmethod(operator.attrgetter("_cache"))
    def _getImage(self, idx):
        return self.pattern2DDet[idx]

    def __len__(self):
        return len(self.selectedList)

class PatternViewer(QtWidgets.QWidget):
    def __init__(self, dataModel, parent=None):
        super().__init__(parent=parent)
        self.rotation = 0
        self.initUI()
        self.updateDataModel(dataModel)
        self.updateRotation(self.rotation)

    def initUI(self):
        grid = QtWidgets.QGridLayout()  
        self.imageViewer = pg.ImageView()
        grid.addWidget(self.imageViewer, 0, 0)

        self.indexGroup = QtWidgets.QGroupBox("Index")
        grid.addWidget(self.indexGroup, 1, 0)
                
        hbox = QtWidgets.QHBoxLayout()
        self.indexGroup.setLayout(hbox)
        self.patternSelectSpinBox = QtWidgets.QSpinBox(self)
        self.patternSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent=self)
        self.patternNumberLabel = QtWidgets.QLabel()
        hbox.addWidget(self.patternSelectSpinBox)
        hbox.addWidget(self.patternNumberLabel)
        hbox.addWidget(self.patternSlider)

        self.rotationSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent=self)
        self.rotationSlider.setMinimum(0)
        self.rotationSlider.setMaximum(359)
        self.rotationSlider.setValue(0)
        self.rotationSlider.valueChanged.connect(self.updateRotation)
        grid.addWidget(self.rotationSlider, 2, 0)

        self.colormapBox = QtWidgets.QComboBox(parent=self)
        self.colormapBox.addItems(plt.colormaps())
        self.colormapBox.currentTextChanged.connect(
            lambda cm: self.imageViewer.setColorMap(pg.colormap.getFromMatplotlib(cm))
        )
        grid.addWidget(self.colormapBox, 3, 0)

        self.setLayout(grid)

    def updateDataModel(self, dataModel):
        self._dm = dataModel # data model
        numData = len(self._dm)
        self.patternSelectSpinBox.setRange(0, numData-1)
        self.patternSelectSpinBox.valueChanged.connect(self._dm.select)
        self._dm.selected.connect(self.updateImage)
        self.patternSlider.setMinimum(0)
        self.patternSlider.setMaximum(numData-1)
        self.patternNumberLabel.setText(f"/{numData}")
        self.patternSlider.setValue(0)
        self.patternSelectSpinBox.valueChanged.connect(self.patternSlider.setValue)
        self.patternSlider.valueChanged.connect(self.patternSelectSpinBox.setValue)

    def updateRotation(self, angle):
        self.rotation = angle
        self.updateImage(*self._dm.getSelection())

    def updateImage(self, rawIndex, img):
        sx, sy = img.shape
        x0, x1, y0, y1 = self._dm.pattern2DDet.detr.frame_extent()
        tr = QtGui.QTransform()
        tr.rotate(self.rotation)
        tr.scale((x1 - x0) / sx, (y1 - y0) / sy)
        tr.translate(x0 - 0.5, y0 - 0.5)
        # print(rawIndex)
        self.indexGroup.setTitle(f"{rawIndex:06d}")
        self.imageViewer.setImage(img, transform=tr, autoRange=False)
