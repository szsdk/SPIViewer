import emcfile as ef
import operator
import numpy as np
import cachetools
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from . import utils

__all__ = [
    "PatternDataModel",
    "PatternViewer",
]


def _symmetrizedDetr(det, symmetrize):
    if not symmetrize:
        new_det = det
    else:
        dd = det.to_dict()
        dd['coor'] = np.concatenate(
            [dd['coor'], dd['coor'] * np.array([-1, -1, 1])], axis=0)
        dd['factor'] = np.concatenate([dd['factor']] * 2)
        dd['mask'] = np.concatenate([dd['mask']] * 2)
        new_det = ef.detector(**dd)
    return ef.det_render(new_det)


class PatternDataModel(QtCore.QObject):
    selected = QtCore.pyqtSignal(int, np.ndarray)
    selectedListChanged = QtCore.pyqtSignal()

    def __init__(self,
                 patterns,
                 detector=None,
                 initIndex=0,
                 selectedList=None,
                 symmetrize = False):
        super().__init__()
        self._initialized = False
        self.patterns = patterns
        self.detector = detector
        self.detectorRender = None
        # self.pattern2DDet = patterns2DDet
        self.symmetrize = False
        self._cache = cachetools.LRUCache(maxsize=32)
        self._idx = initIndex
        self.selectedList = (np.arange(self.patterns.shape[0])
                             if selectedList is None else selectedList)
        self._protectIdx = False
        self.setSymmetrize(symmetrize)
        self._initialized = True

    def setSymmetrize(self, symmetrize):
        if self.symmetrize == symmetrize and self._initialized:
            return
        self.symmetrize = symmetrize
        if self.detector is not None:
            self.detectorRender = _symmetrizedDetr(self.detector, symmetrize)
        self._cache.clear()
        self.select(self.index)

    def updateSelectedList(self, selectedList):
        self.selectedList = (np.arange(self.patterns.shape[0])
                             if selectedList is None else selectedList)
        self.selectedListChanged.emit()
        self.select(0)

    @property
    def index(self):
        return self._idx

    def select(self, idx: int):
        if self._protectIdx:
            return
        self._protectIdx = True
        self._idx = idx
        self.selected.emit(*self.getSelection())
        self._protectIdx = False

    def selectNext(self, d: int = 1):
        self.select((self.index + d) % len(self))

    def selectPrevious(self, d: int = 1):
        self.select((self.index - d) % len(self))

    def selectRandomly(self):
        self.select(np.random.choice(len(self)))

    def getSelection(self):
        rawIndex = int(self.selectedList[self._idx])
        return rawIndex, self._getImage(rawIndex)

    @cachetools.cachedmethod(operator.attrgetter("_cache"))
    def _getImage(self, idx):
        img = self.patterns[idx]
        if not self.symmetrize:
            if self.detectorRender is None:
                return img
            else:
                return self.detectorRender.render(img)
        else:
            if self.detectorRender is None:
                return (img + img[::-1, ::-1]) / 2
            else:
                return self.detectorRender.render(np.concatenate([img] * 2))

    def __len__(self):
        return len(self.selectedList)


class PatternViewer(QtWidgets.QWidget):
    def __init__(self, dataModel, parent=None):
        super().__init__(parent=parent)
        self.rotation = 0
        self.initUI()
        self._dm = dataModel  # data model
        self.patternSelectSpinBox.valueChanged.connect(self._dm.select)
        self.patternSlider.valueChanged.connect(self._dm.select)
        self._dm.selected.connect(
            lambda a, b: self.patternSelectSpinBox.setValue(self._dm.index))
        self._dm.selected.connect(
            lambda a, b: self.patternSlider.setValue(self._dm.index))
        self._dm.selected.connect(self.updateImage)
        self._dm.selectedListChanged.connect(self.updatePatternRange)
        self._imageInit = True
        self.updatePatternRange()
        self.updateRotation(self.rotation)
        self._gears = {
            QtCore.Qt.Key_N: utils.Gear([100, 10, 1], [0.1, 0.5]),
            QtCore.Qt.Key_P: utils.Gear([100, 10, 1], [0.1, 0.5]),
            QtCore.Qt.Key_A: utils.Gear([5, 1], [0.2]),
            QtCore.Qt.Key_S: utils.Gear([5, 1], [0.2]),
        }

    def keyPressEvent(self, event):
        event.accept()
        if event.key() == QtCore.Qt.Key_N:
            self._dm.selectNext(d=self._gears[QtCore.Qt.Key_N].getSpeed())
        elif event.key() == QtCore.Qt.Key_P:
            self._dm.selectPrevious(d=self._gears[QtCore.Qt.Key_N].getSpeed())
        elif event.key() == QtCore.Qt.Key_R:
            self._dm.selectRandomly()
        elif event.key() == QtCore.Qt.Key_A:
            d = self._gears[QtCore.Qt.Key_A].getSpeed()
            self.rotationSlider.setValue(
                (self.rotationSlider.value() - d) % 360)
        elif event.key() == QtCore.Qt.Key_S:
            d = self._gears[QtCore.Qt.Key_S].getSpeed()
            self.rotationSlider.setValue(
                (self.rotationSlider.value() + d) % 360)
        else:
            event.ignore()

    def initUI(self):
        grid = QtWidgets.QGridLayout()
        self.imageViewer = pg.ImageView()
        grid.addWidget(self.imageViewer, 0, 0, 1, 2)

        self.indexGroup = QtWidgets.QGroupBox("Index")
        hbox = QtWidgets.QHBoxLayout()
        self.indexGroup.setLayout(hbox)
        self.patternSelectSpinBox = QtWidgets.QSpinBox(self)
        self.patternSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal,
                                               parent=self)
        self.patternNumberLabel = QtWidgets.QLabel()
        hbox.addWidget(self.patternSelectSpinBox)
        hbox.addWidget(self.patternNumberLabel)
        hbox.addWidget(self.patternSlider)
        grid.addWidget(self.indexGroup, 1, 0, 1, 1)

        self.imageGroup = QtWidgets.QGroupBox("Image")
        igLayout = QtWidgets.QGridLayout()
        self.rotationSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal,
                                                parent=self)
        self.rotationSlider.setMinimum(0)
        self.rotationSlider.setMaximum(359)
        self.rotationSlider.setValue(0)
        self.rotationSlider.valueChanged.connect(self.updateRotation)
        # grid.addWidget(self.rotationSlider, 2, 1)
        # grid.addWidget(QtWidgets.QLabel("rotation"), 2, 0)
        igLayout.addWidget(QtWidgets.QLabel("rotation"), 0, 0)
        igLayout.addWidget(self.rotationSlider, 0, 1, 1, 2)

        self.symmetrizeCheckBox = QtWidgets.QCheckBox("symmetrize")
        self.symmetrizeCheckBox.stateChanged.connect(
            lambda a: self._dm.setSymmetrize(
                self.symmetrizeCheckBox.isChecked()
            )
        )
        igLayout.addWidget(self.symmetrizeCheckBox, 1, 2)

        self.colormapBox = QtWidgets.QComboBox(parent=self)
        self.colormapBox.addItems(plt.colormaps())
        self.colormapBox.currentTextChanged.connect(
            lambda cm: self.imageViewer.setColorMap(
                pg.colormap.getFromMatplotlib(cm)))
        self.colormapBox.setCurrentText("magma")
        self.colormapBox.currentTextChanged.emit("magma")
        igLayout.addWidget(QtWidgets.QLabel("colormap"), 1, 0)
        igLayout.addWidget(self.colormapBox, 1, 1)

        self.imageGroup.setLayout(igLayout)
        grid.addWidget(self.imageGroup, 2, 0)

        # grid.addWidget(QtWidgets.QLabel("colormap"), 3, 0)
        # grid.addWidget(self.colormapBox, 3, 1)

        self.setLayout(grid)

    def updatePatternRange(self):
        numData = len(self._dm)
        self.patternSelectSpinBox.setRange(0, numData - 1)
        self.patternSlider.setMinimum(0)
        self.patternSlider.setMaximum(numData - 1)
        self.patternNumberLabel.setText(f"/{numData}")
        self.patternSlider.setValue(0)

    def updateRotation(self, angle):
        self.rotation = angle
        self.updateImage(*self._dm.getSelection())

    def updateImage(self, rawIndex, img):
        sx, sy = img.shape
        x0, x1, y0, y1 = self._dm.detectorRender.frame_extent()
        tr = QtGui.QTransform()
        tr.rotate(self.rotation)
        tr.scale((x1 - x0) / sx, (y1 - y0) / sy)
        tr.translate(x0 - 0.5, y0 - 0.5)
        s = self._dm.patterns[rawIndex].sum()
        self.indexGroup.setTitle(
            f"index: {rawIndex:06d}/{self._dm.patterns.shape[0]:06d} sum:{s}"
        )
        if self._imageInit:
            self.imageViewer.setImage(img, transform=tr)
            self._imageInit = False
        else:
            self.imageViewer.setImage(
                img,
                transform=tr,
                autoRange=False,
                autoHistogramRange=False,
                autoLevels=False,
            )
