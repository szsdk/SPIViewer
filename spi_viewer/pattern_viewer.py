import logging
from pathlib import Path

import emcfile as ef
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import utils
from ._angular_statistic_viewer import AngularStatisticViewer
from ._pattern_data_model import PatternDataModel

__all__ = [
    "PatternDataModel",
    "PatternViewer",
]

_logger = logging.getLogger(__file__)


class InformationLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._information = dict()

    def update(self, info):
        self._information.update(info)
        self._updateText()

    def _updateText(self):
        text = []
        for k, v in self._information.items():
            if v is None:
                continue
            text.append(f"{k}: {v}")
        self.setText("\n".join(text))
        self.adjustSize()


class PatternViewerShortcuts:
    def __init__(self):
        self._gears = {
            "nextPattern": utils.Gear([100, 10, 1], [0.1, 0.5]),
            "previousPattern": utils.Gear([100, 10, 1], [0.1, 0.5]),
            "left": utils.Gear([5, 1], [0.2]),
            "right": utils.Gear([5, 1], [0.2]),
        }
        self.bookmarks = dict()
        self._marking = False
        self._custom = dict()

    def register(self, k, f):
        self._custom[k] = f

    def keyPressEvent(self, event, pv: "PatternViewer"):
        event.accept()
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self._marking = False
            pv.imageViewer.setFocus()
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
                pv.currentDataset.select(rawIndex)
            pv.rotationSlider.setValue(rotation)
            return

        if text == "m":
            self._marking = True
        elif text == "g":
            pv.patternIndexSpinBox.selectAll()
            pv.patternIndexSpinBox.setFocus()
        elif text == "l":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectNext(
                d=self._gears["nextPattern"].getSpeed())
        elif text == "h":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectPrevious(
                d=self._gears["previousPattern"].getSpeed())
        elif text == "j":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            c = pv.currentDatasetBox
            c.setCurrentIndex((c.currentIndex() + 1) % c.count())
        elif text == "k":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            c = pv.currentDatasetBox
            c.setCurrentIndex((c.currentIndex() - 1) % c.count())
        elif text == "r":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectRandomly()
        elif text == "-":
            pv.setRotation(
                (pv.rotation - self._gears["left"].getSpeed()) % 360)
        elif text == "=":
            pv.setRotation(
                (pv.rotation + self._gears["right"].getSpeed()) % 360)
        elif text == "a":
            pv.dataset2.addPattern(pv.currentDataset.rawIndex)
        elif text == "x":
            pv.currentDataset.removePattern(pv.currentDataset.rawIndex)
        elif text == "S":
            pv.switchDatasets()
        elif text == "s":
            idx = pv.currentDataset.rawIndex
            pv.currentDataset.removePattern(pv.currentDataset.rawIndex)
            pv.dataset2.addPattern(idx)
        elif text in self._custom:
            self._custom[text]()
        else:
            event.ignore()


class PatternViewer(QtWidgets.QMainWindow):
    rotationChanged = QtCore.pyqtSignal(int)
    currentImageChanged = QtCore.pyqtSignal(object)

    def __init__(self, datasets, parent=None):
        super().__init__(parent=parent)
        self._rotation = 0
        self._protectRotation = False
        self.datasets = datasets
        self._transform = None
        self._transformInverted = None
        self._currentImage = None
        self.initUI()
        self._currentDatasetName = self.currentDatasetBox.currentText()
        self._dataset2Name = self.dataset2Box.currentText()
        self._imageInitialized = True
        if len(self.datasets) > 0:
            self._setCurrentDataset(self.currentDatasetBox.currentText())
            self.setRotation(self.rotation)
        self.shortcuts = PatternViewerShortcuts()

        self.angularStatisticViewer = None

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
        self.imageViewer.scene.sigMouseMoved.connect(self.mouseMovedEvent)
        self.infoLabel = InformationLabel(parent=self.imageViewer)
        self.infoLabel.setAlignment(QtCore.Qt.AlignTop)
        self.infoLabel.setStyleSheet("color:#888888;")
        self.infoLabel.move(10, 10)
        self.infoLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid.addWidget(self.imageViewer, 1, 0, 1, 2)

        self.datasetGroup = QtWidgets.QGroupBox("Index")
        hbox = QtWidgets.QHBoxLayout()
        self.datasetGroup.setLayout(hbox)
        self.patternIndexSpinBox = QtWidgets.QSpinBox(self)
        self.patternSlider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, parent=self)
        self.patternNumberLabel = QtWidgets.QLabel()
        self.currentDatasetBox = QtWidgets.QComboBox(parent=self)
        self.currentDatasetBox.addItems(self.datasets.keys())
        self.currentDatasetBox.currentTextChanged.connect(
            self._setCurrentDataset)
        self.patternIndexSpinBox.valueChanged.connect(
            lambda v: self.currentDataset.select(v))
        self.patternSlider.valueChanged.connect(
            lambda v: self.currentDataset.select(v))

        self.dataset2Box = QtWidgets.QComboBox(parent=self)
        self.dataset2Box.addItems(self.datasets.keys())
        self.dataset2Box.currentTextChanged.connect(self._setDataset2)
        hbox.addWidget(self.currentDatasetBox)
        hbox.addWidget(self.patternIndexSpinBox)
        hbox.addWidget(self.patternNumberLabel)
        hbox.addWidget(self.patternSlider)
        hbox.addWidget(self.dataset2Box)
        grid.addWidget(self.datasetGroup, 2, 0, 1, 1)

        self.imageGroup = QtWidgets.QGroupBox("Image")
        igLayout = QtWidgets.QGridLayout()
        self.rotationSlider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, parent=self)
        self.rotationSlider.setMinimum(0)
        self.rotationSlider.setMaximum(359)
        self.rotationSlider.setValue(0)
        self.rotationSlider.valueChanged.connect(self.setRotation)
        self.rotationChanged.connect(self.rotationSlider.setValue)
        self.rotationChanged.connect(
            lambda r: self.setImage(self.currentDataset.getSelection()))

        igLayout.addWidget(QtWidgets.QLabel("rotation"), 0, 0)
        igLayout.addWidget(self.rotationSlider, 0, 1, 1, 3)

        self.symmetrizeCheckBox = QtWidgets.QCheckBox("symmetrize")
        self.symmetrizeCheckBox.stateChanged.connect(
            lambda a: self.currentDataset.setSymmetrize(self.symmetrizeCheckBox
                                                        .isChecked()))
        igLayout.addWidget(self.symmetrizeCheckBox, 1, 2)

        self.applyMaskCheckBox = QtWidgets.QCheckBox("apply mask")
        self.applyMaskCheckBox.stateChanged.connect(
            lambda a: self.currentDataset.setApplyMask(self.applyMaskCheckBox.
                                                       isChecked()))
        igLayout.addWidget(self.applyMaskCheckBox, 1, 3)

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
        grid.addWidget(self.imageGroup, 3, 0)

        self.setCentralWidget(QtWidgets.QWidget(parent=self))
        self.centralWidget().setLayout(grid)

        self.menuBar = self.menuBar()
        self.menuBar.setNativeMenuBar(False)
        fileMenu = self.menuBar.addMenu("&File")
        openAction = fileMenu.addAction("&Open")
        openAction.triggered.connect(lambda: print("NotImplemented"))
        analysisMenu = self.menuBar.addMenu("&Analysis")
        angularStatisticAction = analysisMenu.addAction("Angular statistic")
        angularStatisticAction.triggered.connect(self._angularStatistic)

    def _angularStatistic(self):
        if self.angularStatisticViewer is None:
            self.angularStatisticViewer = AngularStatisticViewer(
                self.currentDataset, bins=51)
            self.currentImageChanged.connect(
                self.angularStatisticViewer.updatePlot)
        self.angularStatisticViewer.show()
        self.currentImageChanged.emit(self)

    def mouseMovedEvent(self, pos):
        if self._currentImage is None:
            return
        mousePoint = self.imageViewer.view.mapSceneToView(pos)
        x, y = self._transformInverted.map(mousePoint.x(), mousePoint.y())
        x_i = round(x)
        y_i = round(y)
        if (x_i >= 0 and x_i < self._currentImage.shape[0] and y_i >= 0
                and y_i < self._currentImage.shape[1]):
            v = self._currentImage[x_i, y_i]
            if np.isfinite(v):
                vstr = f"{v:.3f}"
            else:
                vstr = str(v)
            self.infoLabel.update({
                "position": f"({x:.2f}, {y:.2f})",
                "value": vstr,
            })
        else:
            self.infoLabel.update({"position": None, "value": None})

    def setDataset(self, name, d):
        needUpdate = name in self.datasets
        self.datasets[name] = d
        self.currentDatasetBox.addItem(name)
        self.dataset2Box.addItem(name)
        if needUpdate:
            self._setCurrentDataset(name)

    def removeDataset(self, name):
        if name not in self.datasets:
            return
        index = self.currentDatasetBox.findText(name)  # find the index of text
        self.currentDatasetBox.removeItem(index)
        index = self.dataset2Box.findText(name)  # find the index of text
        self.dataset2Box.removeItem(index)
        if len(self.datasets) == 0:
            raise NotADirectoryError()
        if name == self._currentDatasetName:
            self._setCurrentDataset(next(self.datasets.keys()))

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
            lambda idx: self.patternIndexSpinBox.setValue(idx))
        self.currentDataset.selected.connect(
            lambda idx: self.patternSlider.setValue(idx))
        self.currentDataset.selected.connect(
            lambda idx: self.setImage(self.currentDataset.getSelection()))
        self.currentDataset.selectedListChanged.connect(
            self.updatePatternRange)
        self.updatePatternRange()
        self.currentDataset.select(pidx)
        self.symmetrizeCheckBox.setChecked(self.currentDataset.symmetrize)
        self.applyMaskCheckBox.setChecked(self.currentDataset.applyMask)

    def updatePatternRange(self):
        numData = len(self.currentDataset)
        self.patternIndexSpinBox.setRange(0, numData - 1)
        self.patternSlider.setMinimum(0)
        self.patternSlider.setMaximum(numData - 1)
        self.patternNumberLabel.setText(f"/{numData}")
        self.patternSlider.setValue(0)

    @property
    def rotation(self):
        return self._rotation

    def setRotation(self, r):
        if self._protectRotation:
            return
        self._protectRotation = True
        self._rotation = r
        self.rotationChanged.emit(r)
        self._protectRotation = False

    def setImage(self, img):
        sx, sy = img.shape
        x0, x1, y0, y1 = self.currentDataset.detectorRender.frame_extent()
        tr = QtGui.QTransform()
        tr.rotate(self.rotation)
        tr.scale((x1 - x0) / sx, (y1 - y0) / sy)
        tr.translate(x0 - 0.5, y0 - 0.5)
        self._transform = tr
        self._transformInverted = tr.inverted()[0]
        s = self.currentDataset.patterns[self.currentDataset.rawIndex].sum()
        self.infoLabel.update({
            "dataset": self._currentDatasetName,
            "index":
            f"{self.currentDataset.rawIndex:06d}/{self.currentDataset.patterns.shape[0]:06d}",
            "sum": s,
            "rotation": f"{self.rotationSlider.value()}°",
        })
        self._currentImage = img
        if self._imageInitialized:
            self.imageViewer.setImage(img, transform=tr)
            self._imageInitialized = False
        else:
            self.imageViewer.setImage(
                img,
                transform=tr,
                autoRange=False,
                autoHistogramRange=False,
                autoLevels=False,
            )
        self.currentImageChanged.emit(self)


def patternViewer(src, detector=None):
    if detector is not None:
        pattern = (src if isinstance(src, ef.PatternsSOneEMC) else
                   ef.PatternsSOneEMC(src))
        det = ef.detector(detector)
        datasets = {
            "(default)": PatternDataModel(pattern, detector=det, modify=False)
        }
    else:
        datasets = {
            k: v if isinstance(v, PatternDataModel) else PatternDataModel(**v)
            for k, v in src.items()
        }

    return PatternViewer(datasets)
