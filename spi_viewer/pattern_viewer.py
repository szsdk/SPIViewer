import emcfile as ef
import operator
import numpy as np
import cachetools
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from . import utils
import logging

__all__ = [
    "PatternDataModel",
    "PatternViewer",
]

_logger = logging.getLogger(__file__)


def _symmetrizedDetr(det, symmetrize):
    if not symmetrize:
        new_det = det
    else:
        dd = det.to_dict()
        dd["coor"] = np.concatenate(
            [dd["coor"], dd["coor"] * np.array([-1, -1, 1])], axis=0
        )
        dd["factor"] = np.concatenate([dd["factor"]] * 2)
        dd["mask"] = np.concatenate([dd["mask"]] * 2)
        new_det = ef.detector(**dd)
    return ef.det_render(new_det)


class PatternDataModel(QtCore.QObject):
    selected = QtCore.pyqtSignal(np.ndarray)
    selectedListChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        patterns,
        detector=None,
        initIndex=0,
        selectedList=None,
        symmetrize=False,
        applyMask=False,
        modify=True
    ):
        super().__init__()
        self._initialized = False
        self.patterns = patterns
        self.detector = detector
        self.detectorRender = None
        self.symmetrize = False
        self._cache = cachetools.LRUCache(maxsize=32)
        self._index = initIndex
        self.modify = modify
        self.selectedList = (
            np.arange(self.patterns.shape[0]) if selectedList is None else selectedList
        )
        self._rawIndex = int(self.selectedList[self.index])
        self._applyMask = applyMask
        self._protectIndex = False
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

    def setSelectedList(self, selectedList):
        if not self.modify:
            raise Exception("Cannot modify")
        self.selectedList = selectedList
        self.selectedListChanged.emit()
        self.select(0)

    def addPattern(self, rawIndex):
        if rawIndex in self.selectedList:
            return
        newSelectedList = np.sort(np.append(self.selectedList, rawIndex))
        self.setSelectedList(newSelectedList)

    def removePattern(self, rawIndex):
        if rawIndex not in self.selectedList:
            return
        i = np.where(self.selectedList == rawIndex)[0][0]
        newSelectedList = np.delete(self.selectedList, i)
        self.setSelectedList(newSelectedList)
        self.select(min(i, len(newSelectedList) - 1))

    @property
    def applyMask(self):
        return self._applyMask

    def setApplyMask(self, applyMask: bool):
        update = self._applyMask != applyMask
        self._applyMask = applyMask
        if update:
            self._cache.clear()
            self.select(self.index)

    @property
    def index(self):
        return self._index

    @property
    def rawIndex(self):
        return self._rawIndex

    def select(self, index: int):
        if self._protectIndex:
            return
        self._index = index
        self.selectByRawIndex(int(self.selectedList[self.index]))

    def selectByRawIndex(self, rawIndex):
        self._rawIndex = rawIndex
        self._protectIndex = True
        self.selected.emit(self.getSelection())
        self._protectIndex = False

    def selectNext(self, d: int = 1):
        self.select((self.index + d) % len(self))

    def selectPrevious(self, d: int = 1):
        self.select((self.index - d) % len(self))

    def selectRandomly(self):
        self.select(np.random.choice(len(self)))

    def getSelection(self):
        return self.getImage(self.rawIndex)

    @cachetools.cachedmethod(operator.attrgetter("_cache"))
    def getImage(self, index):
        img = self.patterns[index]
        if not self.symmetrize:
            if self.detectorRender is None:
                ans = img
            else:
                ans = self.detectorRender.render(img)
        else:
            if self.detectorRender is None:
                ans = (img + img[::-1, ::-1]) / 2
            else:
                ans = self.detectorRender.render(
                    np.concatenate([img] * 2), intensity=True
                )
        if self.applyMask and hasattr(ans, "mask"):
            ans[ans.mask] = np.nan
        return ans

    def __len__(self):
        return len(self.selectedList)


class PatternViewerShortcuts:
    def __init__(self):
        self._gears = {
            "n": utils.Gear([100, 10, 1], [0.1, 0.5]),
            "p": utils.Gear([100, 10, 1], [0.1, 0.5]),
            "-": utils.Gear([5, 1], [0.2]),
            "=": utils.Gear([5, 1], [0.2]),
        }
        self.bookmarks = dict()
        self._marking = False
        self._custom = dict()

    def register(self, k, f):
        self._custom[k] = f

    def keyPressEvent(self, event, pv):
        event.accept()
        text = event.text()
        if self._marking:
            self._marking = False
            if not text.isdigit():
                raise Exception()
            self.bookmarks[text] = pv.currentDataset.rawIndex, pv.rotation
            return

        if text.isdigit():
            rawIndex, rotation = self.bookmarks.get(text, (None, None))
            if rawIndex is None:
                raise Exception()
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            if rawIndex not in pv.currentDataset.selectedList:
                pv.currentDataset.selectByRawIndex(rawIndex)
            else:
                index = np.where(pv.currentDataset.selectedList == rawIndex)[0][0]
                pv.currentDataset.select(rawIndex)
            pv.rotationSlider.setValue(rotation)
            return

        key = event.key()
        if text == "m":
            self._marking = True
        elif text == "n":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectNext(d=self._gears["n"].getSpeed())
        elif text == "p":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectPrevious(d=self._gears["p"].getSpeed())
        elif text == "r":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectRandomly()
        elif text == "-":
            d = self._gears["-"].getSpeed()
            pv.rotationSlider.setValue((pv.rotationSlider.value() - d) % 360)
        elif text == "=":
            d = self._gears["="].getSpeed()
            pv.rotationSlider.setValue((pv.rotationSlider.value() + d) % 360)
        elif text  == "a":
            pv.dataset2.addPattern(pv.currentDataset.rawIndex)
        elif text  == "x":
            pv.currentDataset.removePattern(pv.currentDataset.rawIndex)
        elif text  == "S":
            pv.switchDatasets()
        elif text  == "s":
            idx = pv.currentDataset.rawIndex
            pv.currentDataset.removePattern(pv.currentDataset.rawIndex)
            pv.dataset2.addPattern(idx)
        elif text in self._custom:
            self._custom[text]()
        else:
            event.ignore()


class PatternViewer(QtWidgets.QMainWindow):
    def __init__(self, datasets, parent=None):
        super().__init__(parent=parent)
        self.rotation = 0
        self.datasets = datasets
        self.initUI()
        self._currentDatasetName = self.currentDatasetBox.currentText()
        self._dataset2Name = self.dataset2Box.currentText()
        self._imageInit = True
        if len(self.datasets) > 0:
            self._setCurrentDataset(self.currentDatasetBox.currentText())
            self.updateRotation(self.rotation)
        self.shortcuts = PatternViewerShortcuts()

    @property
    def currentDataset(self):
        return self.datasets[self._currentDatasetName]

    @property
    def dataset2(self):
        return self.datasets[self._dataset2Name]

    def keyPressEvent(self, event):
        self.shortcuts.keyPressEvent(event, self)

    def initUI(self):
        grid = QtWidgets.QGridLayout()

        self.imageViewer = pg.ImageView()
        self.infoLabel = QtWidgets.QLabel("info", parent=self.imageViewer)
        self.infoLabel.setAlignment(QtCore.Qt.AlignTop)
        self.infoLabel.setStyleSheet("color:#888888;")
        self.infoLabel.move(10, 10)
        grid.addWidget(self.imageViewer, 0, 0, 1, 2)

        self.indexGroup = QtWidgets.QGroupBox("Index")
        hbox = QtWidgets.QHBoxLayout()
        self.indexGroup.setLayout(hbox)
        self.patternSelectSpinBox = QtWidgets.QSpinBox(self)
        self.patternSlider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, parent=self
        )
        self.patternNumberLabel = QtWidgets.QLabel()
        self.currentDatasetBox = QtWidgets.QComboBox(parent=self)
        self.currentDatasetBox.addItems(self.datasets.keys())
        self.currentDatasetBox.currentTextChanged.connect(self._setCurrentDataset)
        self.patternSelectSpinBox.valueChanged.connect(lambda v: self.currentDataset.select(v))
        self.patternSlider.valueChanged.connect(lambda v: self.currentDataset.select(v))

        self.dataset2Box = QtWidgets.QComboBox(parent=self)
        self.dataset2Box.addItems(self.datasets.keys())
        self.dataset2Box.currentTextChanged.connect(self._setDataset2)
        hbox.addWidget(self.currentDatasetBox)
        hbox.addWidget(self.patternSelectSpinBox)
        hbox.addWidget(self.patternNumberLabel)
        hbox.addWidget(self.patternSlider)
        hbox.addWidget(self.dataset2Box)
        grid.addWidget(self.indexGroup, 1, 0, 1, 1)

        self.imageGroup = QtWidgets.QGroupBox("Image")
        igLayout = QtWidgets.QGridLayout()
        self.rotationSlider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, parent=self
        )
        self.rotationSlider.setMinimum(0)
        self.rotationSlider.setMaximum(359)
        self.rotationSlider.setValue(0)
        self.rotationSlider.valueChanged.connect(self.updateRotation)
        igLayout.addWidget(QtWidgets.QLabel("rotation"), 0, 0)
        igLayout.addWidget(self.rotationSlider, 0, 1, 1, 3)

        self.symmetrizeCheckBox = QtWidgets.QCheckBox("symmetrize")
        self.symmetrizeCheckBox.stateChanged.connect(
            lambda a: self.currentDataset.setSymmetrize(self.symmetrizeCheckBox.isChecked())
        )
        igLayout.addWidget(self.symmetrizeCheckBox, 1, 2)

        self.applyMaskCheckBox = QtWidgets.QCheckBox("apply mask")
        self.applyMaskCheckBox.stateChanged.connect(
            lambda a: self.currentDataset.setApplyMask(self.applyMaskCheckBox.isChecked())
        )
        igLayout.addWidget(self.applyMaskCheckBox, 1, 3)

        self.colormapBox = QtWidgets.QComboBox(parent=self)
        self.colormapBox.addItems(plt.colormaps())
        self.colormapBox.currentTextChanged.connect(
            lambda cm: self.imageViewer.setColorMap(pg.colormap.getFromMatplotlib(cm))
        )
        self.colormapBox.setCurrentText("magma")
        self.colormapBox.currentTextChanged.emit("magma")
        igLayout.addWidget(QtWidgets.QLabel("colormap"), 1, 0)
        igLayout.addWidget(self.colormapBox, 1, 1)

        self.imageGroup.setLayout(igLayout)
        grid.addWidget(self.imageGroup, 2, 0)

        self.setCentralWidget(QtWidgets.QWidget(parent=self))
        self.centralWidget().setLayout(grid)
        # self.setLayout(grid)

        self.menuBar = self.menuBar()
        self.menuBar.setNativeMenuBar(False)
        fileMenu = self.menuBar.addMenu("&File")
        openAction = fileMenu.addAction("&Open")
        openAction.triggered.connect(lambda : print("NotImplemented"))


    def switchDatasets(self):
        t = self._dataset2Name
        self.dataset2Box.setCurrentText(self._currentDatasetName)
        self.currentDatasetBox.setCurrentText(t)

    def _setDataset2(self, sl: str):
        self._dataset2Name = sl

    def _setCurrentDataset(self, sl: str):
        try:
            self.currentDataset.selectedListChanged.disconnect()
            self.currentDataset.selected.disconnect()
        except:
            pass
        self._currentDatasetName = sl
        pidx = self.currentDataset.index
        self.currentDataset.selected.connect(
            lambda a: self.patternSelectSpinBox.setValue(self.currentDataset.index)
        )
        self.currentDataset.selected.connect(lambda a: self.patternSlider.setValue(self.currentDataset.index))
        self.currentDataset.selected.connect(self.updateImage)
        self.currentDataset.selectedListChanged.connect(self.updatePatternRange)
        self.updatePatternRange()
        self.currentDataset.select(pidx)
        self.symmetrizeCheckBox.setChecked(self.currentDataset.symmetrize)
        self.applyMaskCheckBox.setChecked(self.currentDataset.applyMask)

    def updatePatternRange(self):
        numData = len(self.currentDataset)
        self.patternSelectSpinBox.setRange(0, numData - 1)
        self.patternSlider.setMinimum(0)
        self.patternSlider.setMaximum(numData - 1)
        self.patternNumberLabel.setText(f"/{numData}")
        self.patternSlider.setValue(0)

    def updateRotation(self, angle):
        self.rotation = angle
        self.updateImage(self.currentDataset.getSelection())

    def updateImage(self, img):
        sx, sy = img.shape
        x0, x1, y0, y1 = self.currentDataset.detectorRender.frame_extent()
        tr = QtGui.QTransform()
        tr.rotate(self.rotation)
        tr.scale((x1 - x0) / sx, (y1 - y0) / sy)
        tr.translate(x0 - 0.5, y0 - 0.5)
        s = self.currentDataset.patterns[self.currentDataset.rawIndex].sum()
        self.indexGroup.setTitle(
            f"index: {self.currentDataset.rawIndex:06d}/{self.currentDataset.patterns.shape[0]:06d} sum: {s}"
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
