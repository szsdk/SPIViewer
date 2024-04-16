import logging
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import emcfile as ef
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtWidgets import QInputDialog, QMessageBox

from . import utils
from ._ang_binned_statistic import ang_binned_statistic
from ._angular_statistic_viewer import AngularStatisticViewer
from ._pattern_data_model import (
    NullPatternDataModel,
    PatternDataModel,
    PatternDataModelBase,
)

__all__ = [
    "PatternViewer",
]

_logger = logging.getLogger(__file__)

__doc__ = """
# SPI Viewer
- D0: the main dataset
- D1: the sencondary dataset

## Shortcuts
- `g`: go to a pattern by typing its index
- `h`, `j`: scroll datasets
- `k`, `l`: scroll patterns
- `-`, `=`: rotate
- `m1`, ..., `m9`: make a bookmark
- `1`, ..., `9`: go to a bookmark
- `0`: a special bookmark for the last shown pattern
- `S`: switch datasets
- `s`: switch a pattern from D0 to D1
- `a`: add a pattern from D0 to D1
- `x`: delete a pattern from D0
- `A`: add a new dataset
- `D`: delete D0
- `e`: edit the pattern indices of D0 with `EDITOR`, `VISUAL` or `vim
- `esc`: reset focus
- `esc`×2: exit
- `?`: show help

## pre-defined functions:
- `ang_stat(img, bins=50, statistic="min", mask=[ef.PixelType.BAD])`:
    Calculate the angular statistic of an image.
"""


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


def edit_with_vim(a, suffix=""):
    if "EDITOR" in os.environ:
        editor = os.environ["EDITOR"]
    elif "VISUAL" in os.environ:
        editor = os.environ["VISUAL"]
    else:
        editor = "vim"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        np.savetxt(temp_file.name, a, fmt="%d")
        subprocess.call(f"{editor} {temp_file.name}", shell=True)
        return np.loadtxt(temp_file.name, dtype=int)


class PatternViewerShortcuts:
    def __init__(self):
        self._gears = {
            "nextPattern": utils.Gear([100, 10, 1], [0.1, 0.2]),
            "previousPattern": utils.Gear([100, 10, 1], [0.1, 0.2]),
            "left": utils.Gear([5, 1], [0.2]),
            "right": utils.Gear([5, 1], [0.2]),
            "close": utils.Gear([True, False], [0.2]),
        }
        self.bookmarks = dict()
        self._marking = False
        self._custom = dict()

    def register(self, k, f):
        self._custom[k] = f

    def unregister(self, k):
        if k in self._custom:
            del self._custom[k]

    def keyPressEvent(self, event, pv: "PatternViewer"):
        event.accept()
        key = event.key()
        if key == QtCore.Qt.Key.Key_Escape:
            self._marking = False
            if self._gears["close"].getSpeed():
                QtWidgets.QApplication.quit()
            else:
                pv.imageViewer.setFocus()
        text = event.text()
        currentDataset = pv.datasetsManager.dataset1
        if self._marking:
            self._marking = False
            if not text.isdigit():
                raise Exception()
            self.bookmarks[text] = currentDataset.rawIndex, pv.rotation
            return

        if text.isdigit():
            rawIndex, rotation = self.bookmarks.get(text, (None, None))
            if rawIndex is None:
                raise Exception()
            self.bookmarks["0"] = currentDataset.rawIndex, pv.rotation
            if rawIndex not in currentDataset.selectedList:
                pv.datasetsManager.selectByRawIndex(rawIndex)
            else:
                pv.datasetsManager.select(rawIndex)
            pv.rotationSlider.setValue(rotation)
            return

        if text == "m":
            self._marking = True
        elif text == "e":
            if currentDataset.modify:
                pv.datasetsManager.setSelectedList(
                    edit_with_vim(currentDataset.selectedList)
                )
            else:
                QMessageBox.warning(
                    pv,
                    "Cannot modify",
                    f"The dataset {pv.datasetsManager.nameOfDataset1} cannot be modified.",
                    QMessageBox.StandardButton.Ok,
                )
        elif text == "g":
            pv.patternIndexSpinBox.selectAll()
            pv.patternIndexSpinBox.setFocus()
        elif text == "l":
            self.bookmarks["0"] = currentDataset.rawIndex, pv.rotation
            pv.datasetsManager.selectNext(d=self._gears["nextPattern"].getSpeed())
        elif text == "h":
            self.bookmarks["0"] = currentDataset.rawIndex, pv.rotation
            pv.datasetsManager.selectPrevious(
                d=self._gears["previousPattern"].getSpeed()
            )
        elif text == "j":
            self.bookmarks["0"] = currentDataset.rawIndex, pv.rotation
            c = pv.currentDatasetBox
            c.setCurrentIndex((c.currentIndex() + 1) % c.count())
        elif text == "k":
            self.bookmarks["0"] = currentDataset.rawIndex, pv.rotation
            c = pv.currentDatasetBox
            c.setCurrentIndex((c.currentIndex() - 1) % c.count())
        elif text == "r":
            self.bookmarks["0"] = currentDataset.rawIndex, pv.rotation
            pv.datasetsManager.selectRandomly()
        elif text == "-":
            pv.setRotation((pv.rotation - self._gears["left"].getSpeed()) % 360)
        elif text == "=":
            pv.setRotation((pv.rotation + self._gears["right"].getSpeed()) % 360)
        elif text == "a":
            pv.datasetsManager.dataset2.addPattern(currentDataset.rawIndex)
        elif text == "x":
            pv.datasetsManager.dataset1.removePattern(currentDataset.rawIndex)
        elif text == "S":
            pv.datasetsManager.switchDatasets()
        elif text == "s":
            idx = currentDataset.rawIndex
            currentDataset.removePattern(currentDataset.rawIndex)
            pv.dataset2.addPattern(idx)
        elif text == "D":
            ret = QMessageBox.question(
                pv,
                "MessageBox",
                f"Are you sure you want to delete the dataset [{pv.datasetsManager.nameOfDataset1}]?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret == QMessageBox.StandardButton.Yes:
                pv.datasetsManager.deleteDataset(pv.datasetsManager.nameOfDataset1)
        elif text == "A":
            newDatasetName, ok = QInputDialog.getText(
                pv, "Text Input Dialog", "Enter your name:"
            )
            if ok:
                pv.datasetsManager.addDataset(
                    newDatasetName,
                    PatternDataModel(
                        currentDataset.patterns,
                        detector=currentDataset.detector,
                        initIndex=None,
                        selectedList=[],
                    ),
                )
        elif text == "?":
            pv.showHelp()
        elif text in self._custom:
            self._custom[text]()
        elif key in self._custom:
            self._custom[key]()
        else:
            event.ignore()


nullPatternDataModel = NullPatternDataModel()


class HelpWindow(QtWidgets.QMainWindow):
    def __init__(self, help_text, parent=None):
        super().__init__(parent=parent)

        self.setWindowFlag(QtCore.Qt.WindowType.Tool)
        self.setWindowTitle("Help")
        self.setGeometry(100, 100, 400, 300)

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.setCentralWidget(self.text_edit)

        self.display_help(help_text)

    def display_help(self, help_text):
        self.text_edit.setMarkdown(help_text)


class CentralCrossROI(pg.ROI):
    def __init__(self, **args):
        line0 = pg.LineSegmentROI([(0, 5), (10, 5)], movable=False)
        for handler in line0.handles:
            handler["item"].hide()
        line1 = pg.LineSegmentROI([(5, 0), (5, 10)], movable=False)
        for handler in line1.handles:
            handler["item"].hide()
        self.line0 = line0
        self.line1 = line1
        pg.ROI.__init__(self, [-5, -5], [10, 10], **args)

    def getState(self):
        return {
            "line0": self.line0.getState(),
            "line1": self.line1.getState(),
        }

    def saveState(self):
        return {
            "line0": self.line0.saveState(),
            "line1": self.line1.saveState(),
        }

    def setState(self, state):
        self.line0.setState(state["line0"])
        self.line1.setState(state["line1"])

    def paint(self, p, *args):
        self.line0.paint(p)
        self.line1.paint(p)


@lru_cache
def _radius_from_shape(shape):
    coor = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        indexing="ij",
    )
    return np.linalg.norm([coor[i] - shape[i] / 2.0 for i in range(2)], axis=0).ravel()


def ang_stat(img, bins=50, statistic="min", mask=[ef.PixelType.BAD], fill_nan=0):
    """
    Calculate the angular statistic of an image.
    """
    a = ang_binned_statistic(
        img,
        _radius_from_shape(img.shape),
        bins=bins,
        statistic=statistic,
    )
    s = a.statistic
    s[np.isnan(s)] = fill_nan
    return np.interp(a.radius, (a.bin_edges[:-1] + a.bin_edges[1:]) / 2, s).reshape(
        img.shape
    )


class DatasetsManager(QtCore.QObject):
    datasetsChanged = QtCore.pyqtSignal(list)
    dataset1Changed = QtCore.pyqtSignal(str)
    dataset2Changed = QtCore.pyqtSignal(str)
    selected = QtCore.pyqtSignal(int)
    selectedListChanged = QtCore.pyqtSignal()

    def __init__(self, datasets, parent=None):
        super().__init__(parent=parent)
        self._datasets = datasets
        self._nameOfDataset1 = None
        self._nameOfDataset2 = None
        self._protectIndex = False

    @property
    def datasets(self):
        return self._datasets

    @property
    def dataset1(self) -> PatternDataModelBase:
        return self.datasets.get(self._nameOfDataset1, nullPatternDataModel)

    @property
    def dataset2(self) -> PatternDataModelBase:
        return self.datasets.get(self._nameOfDataset2, nullPatternDataModel)

    @property
    def nameOfDataset1(self):
        return self._nameOfDataset1

    @property
    def nameOfDataset2(self):
        return self._nameOfDataset2

    def setDataset1ByName(self, name):
        if name not in self._datasets:
            raise Exception(f"{name} is not in the datasets")
        self._nameOfDataset1 = name
        self.dataset1Changed.emit(name)

    def setDataset2ByName(self, name):
        if name not in self._datasets:
            raise Exception(f"{name} is not in the datasets")
        self._nameOfDataset2 = name
        self.dataset2Changed.emit(name)

    def addDataset(self, name, dataset):
        if name == "":
            raise RuntimeError("The name of the dataset cannot be empty")
        if name in self.datasets:
            raise RuntimeError(f"{name} already exists")
        self._datasets[name] = dataset
        self.datasetsChanged.emit(list(self.datasets.keys()))

    def deleteDataset(self, name):
        if name not in self.datasets:
            raise RuntimeError(f"{name} does not exist")
        del self._datasets[name]
        self.datasetsChanged.emit(list(self.datasets.keys()))
        if self.nameOfDataset1 == name:
            self._nameOfDataset1 = (
                list(self.datasets.keys())[0] if len(self.datasets) > 0 else ""
            )
            self.dataset1Changed.emit(self._nameOfDataset1)
        if self._nameOfDataset2 == name:
            self._nameOfDataset2 = (
                list(self.datasets.keys())[0] if len(self.datasets) > 0 else ""
            )
            self.dataset2Changed.emit(self._nameOfDataset2)

    def switchDatasets(self):
        self._nameOfDataset1, self._nameOfDataset2 = (
            self._nameOfDataset2,
            self._nameOfDataset1,
        )
        self.dataset1Changed.emit(self._nameOfDataset1)
        self.dataset2Changed.emit(self._nameOfDataset2)

    def setApplyMask(self, applyMask: bool):
        self.dataset1.setApplyMask(applyMask)

    def setSymmetrize(self, symmetrize: bool):
        self.dataset1.setSymmetrize(symmetrize)

    def select(self, i):
        self.selected.emit(self.dataset1.select(i))

    def selectByRawIndex(self, i):
        self._protectIndex = True
        self.dataset1.selectByRawIndex(i)
        self.selected.emit(self.dataset1.index)
        self._protectIndex = False

    def selectNext(self, d: int = 1):
        self.selected.emit(self.dataset1.selectNext(d))

    def selectPrevious(self, d: int = 1):
        self.selected.emit(self.dataset1.selectPrevious(d))

    def selectRandomly(self):
        self.selected.emit(self.dataset1.selectRandomly())


class _ColorbarManager:
    def __init__(self, imageViewer, name):
        self.imageViewer = imageViewer
        self._currentDataset = name
        self._data = dict()

    def update(self, name):
        self._data[self._currentDataset] = {
            "levels": self.imageViewer.ui.histogram.getLevels(),
            # "range": self.imageViewer.ui.histogram.getHistogramRange(),
        }
        self._currentDataset = name
        if name in self._data:
            data = self._data[name]
            self.imageViewer.ui.histogram.setLevels(*data["levels"])
            # self.imageViewer.ui.histogram.setHistogramRange(data["range"])


class PatternViewer(QtWidgets.QMainWindow):
    rotationChanged = QtCore.pyqtSignal(int)
    currentImageChanged = QtCore.pyqtSignal(object)

    def __init__(self, datasets, separateColorbar: bool = False, parent=None):
        super().__init__(parent=parent)
        self._rotation = 0
        self._protectRotation = False
        self._imageInitialized = False
        # self.datasets = datasets
        self.datasetsManager = DatasetsManager(datasets, parent=self)
        self._transform = None
        self._transformInverted = None
        self._currentImage = None
        self.menus = dict()
        self.initUI()
        self.datasetsManager.datasetsChanged.connect(self.datasetsChanged)
        self.datasetsManager.dataset1Changed.connect(self._setCurrentDataset)
        self.datasetsManager.dataset1Changed.connect(
            self.currentDatasetBox.setCurrentText
        )
        self.datasetsManager.dataset2Changed.connect(self.dataset2Box.setCurrentText)
        self.datasetsManager.setDataset1ByName(self.currentDatasetBox.currentText())
        self.datasetsManager.setDataset2ByName(self.dataset2Box.currentText())
        if len(self.datasetsManager.datasets) > 0:
            self.datasetsManager.setDataset1ByName(self.currentDatasetBox.currentText())
            self.setRotation(self.rotation)
        self.shortcuts = PatternViewerShortcuts()

        self.angularStatisticViewer = None
        self.currentImageChangedFunc = None
        # This is a shortcut function which would be called whenever the image is changed.
        # It could be modified directly. Setting it to `None` avoids the calling.
        self.currentImageChanged.connect(self._callCurrentImageChangedFunc)

        self._colorbarManager = _ColorbarManager(
            self.imageViewer, self.datasetsManager.nameOfDataset1
        )
        if separateColorbar:
            self.datasetsManager.dataset1Changed.connect(self._colorbarManager.update)

    def datasetsChanged(self, datasetNames):
        self.currentDatasetBox.currentTextChanged.disconnect(
            self.datasetsManager.setDataset1ByName
        )
        self.currentDatasetBox.clear()
        self.currentDatasetBox.addItems(datasetNames)
        self.currentDatasetBox.currentTextChanged.connect(
            self.datasetsManager.setDataset1ByName
        )

        self.dataset2Box.currentTextChanged.disconnect(
            self.datasetsManager.setDataset2ByName
        )
        self.dataset2Box.clear()
        self.dataset2Box.addItems(datasetNames)
        self.dataset2Box.currentTextChanged.connect(
            self.datasetsManager.setDataset2ByName
        )

    def _callCurrentImageChangedFunc(self):
        if self.currentImageChangedFunc is not None:
            self.currentImageChangedFunc(self)

    @property
    def currentDatasetName(self):
        return self.datasetsManager.nameOfDataset1

    @property
    def dataset2(self):
        return self.datasets[self.datasetsManager.dataset2]

    # @property
    # def currentDataset(self):
    #     return self.datasetsManager.dataset1

    def keyPressEvent(self, event):
        self.shortcuts.keyPressEvent(event, self)

    def initUI(self):
        grid = QtWidgets.QGridLayout()

        self.imageViewer = pg.ImageView()
        self.imageViewer.scene.sigMouseMoved.connect(self.mouseMovedEvent)
        self.infoLabel = InformationLabel(parent=self.imageViewer)
        self.infoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.infoLabel.setStyleSheet("color:#888888;")
        self.infoLabel.move(10, 10)
        self.infoLabel.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(self.imageViewer, 1, 0, 1, 2)

        self.datasetsManager.selected.connect(self._setPatternIndex)
        self.datasetsManager.selected.connect(self.updateImage)
        self.datasetsManager.selectedListChanged.connect(self.updatePatternRange)

        self.datasetGroup = QtWidgets.QGroupBox("Index")
        hbox = QtWidgets.QHBoxLayout()
        self.datasetGroup.setLayout(hbox)
        self.patternIndexSpinBox = QtWidgets.QSpinBox(self)
        self.patternSlider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, parent=self
        )
        self.patternNumberLabel = QtWidgets.QLabel()
        self.currentDatasetBox = QtWidgets.QComboBox(parent=self)
        self.currentDatasetBox.addItems(self.datasetsManager.datasets.keys())
        self.currentDatasetBox.currentTextChanged.connect(
            self.datasetsManager.setDataset1ByName
        )
        self.patternIndexSpinBox.valueChanged.connect(self.datasetsManager.select)
        self.patternSlider.valueChanged.connect(self.datasetsManager.select)

        self.dataset2Box = QtWidgets.QComboBox(parent=self)
        self.dataset2Box.addItems(self.datasetsManager.datasets.keys())
        self.dataset2Box.currentTextChanged.connect(
            self.datasetsManager.setDataset2ByName
        )
        hbox.addWidget(self.currentDatasetBox)
        hbox.addWidget(self.patternIndexSpinBox)
        hbox.addWidget(self.patternNumberLabel)
        hbox.addWidget(self.patternSlider)
        hbox.addWidget(self.dataset2Box)
        grid.addWidget(self.datasetGroup, 2, 0, 1, 1)
        self.setCentralWidget(QtWidgets.QWidget(parent=self))
        self.centralWidget().setLayout(grid)
        self._initImageControlWindow()
        self._initMenuBar()

    def _initImageControlWindow(self):
        self.imageControlWindow = QtWidgets.QWidget()
        self.imageControlWindow.setWindowFlag(QtCore.Qt.WindowType.Tool)
        igLayout = QtWidgets.QGridLayout()
        self.rotationSlider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, parent=self
        )
        self.rotationSlider.setMinimum(0)
        self.rotationSlider.setMaximum(359)
        self.rotationSlider.setValue(0)
        self.rotationSlider.valueChanged.connect(self.setRotation)
        self.rotationChanged.connect(self.rotationSlider.setValue)
        self.rotationChanged.connect(
            lambda r: self.setImage(self.datasetsManager.dataset1.getSelectedImage())
        )

        igLayout.addWidget(QtWidgets.QLabel("rotation"), 0, 0)
        igLayout.addWidget(self.rotationSlider, 0, 1, 1, 3)

        self.flipCheckBox = QtWidgets.QCheckBox("flip")
        self.flipCheckBox.stateChanged.connect(
            lambda state: self.setImage(self._currentImage)
        )
        self.symmetrizeCheckBox = QtWidgets.QCheckBox("symmetrize")
        self.symmetrizeCheckBox.stateChanged.connect(
            lambda a: self.datasetsManager.setSymmetrize(
                self.symmetrizeCheckBox.isChecked()
            )
        )
        self.symmetrizeCheckBox.stateChanged.connect(self.updateImage)
        igLayout.addWidget(self.flipCheckBox, 3, 2)
        igLayout.addWidget(self.symmetrizeCheckBox, 3, 1)

        self.applyMaskCheckBox = QtWidgets.QCheckBox("apply mask")
        self.applyMaskCheckBox.stateChanged.connect(
            lambda a: self.datasetsManager.setApplyMask(
                self.applyMaskCheckBox.isChecked()
            )
        )
        self.applyMaskCheckBox.stateChanged.connect(self.updateImage)
        igLayout.addWidget(self.applyMaskCheckBox, 3, 3)

        self.showCircleCheckBox = QtWidgets.QCheckBox("circle")
        self.cicrleROI = pg.CircleROI((-10, -10), 20)
        self.cicrleROI.hide()
        self.imageViewer.view.addItem(self.cicrleROI)
        self.showCircleCheckBox.stateChanged.connect(
            lambda s: self.cicrleROI.show()
            if self.showCircleCheckBox.isChecked()
            else self.cicrleROI.hide()
        )
        igLayout.addWidget(QtWidgets.QLabel("tools"), 4, 0)
        igLayout.addWidget(self.showCircleCheckBox, 4, 1)

        self.showCrossCheckBox = QtWidgets.QCheckBox("cross")
        self.crossROI = CentralCrossROI()
        self.crossROI.hide()
        self.imageViewer.view.addItem(self.crossROI)
        self.showCrossCheckBox.stateChanged.connect(
            lambda s: self.crossROI.show()
            if self.showCrossCheckBox.isChecked()
            else self.crossROI.hide()
        )
        igLayout.addWidget(self.showCrossCheckBox, 4, 2)

        self.imageViewer.setColorMap(pg.colormap.getFromMatplotlib("magma"))

        # self.applyImageFuncBox = QtWidgets.QLineEdit(parent=self.imageControlWindow)
        self.applyImageFuncBox = QtWidgets.QLineEdit()
        # self.applyImageFuncBox.setPlaceholderText("np.log(x); ang_stat(x)")
        self.applyImageFuncBox.returnPressed.connect(
            lambda: self.setImage(self._currentImage)
        )
        self.applyImageFuncComboBox = QtWidgets.QComboBox(self.imageControlWindow)
        self.applyImageFuncComboBox.addItems(
            ["x", "x ** 0.2", "np.log(x)", "x-ang_stat(x, statistic=np.nanmean)"]
        )
        self.applyImageFuncComboBox.setLineEdit(self.applyImageFuncBox)
        self.applyImageFuncComboBox.currentTextChanged.connect(
            lambda: self.setImage(self._currentImage)
        )
        igLayout.addWidget(QtWidgets.QLabel("apply"), 2, 0)
        igLayout.addWidget(self.applyImageFuncComboBox, 2, 1, 1, 3)

        self.imageControlWindow.setLayout(igLayout)

    def _initMenuBar(self):
        self.menuBar = self.menuBar()
        self.menuBar.setNativeMenuBar(False)
        fileMenu = self.menuBar.addMenu("&File")
        openAction = fileMenu.addAction("&Open")
        openAction.triggered.connect(lambda: print("NotImplemented"))
        saveAction = fileMenu.addAction("&Save")
        saveAction.triggered.connect(self._save)
        saveAction.setShortcut("Ctrl+S")

        analysisMenu = self.menuBar.addMenu("&Analysis")
        angularStatisticAction = analysisMenu.addAction("Angular statistic")
        angularStatisticAction.triggered.connect(self._angularStatistic)
        addSumDatasetAction = analysisMenu.addAction("Sum current dataset")
        addSumDatasetAction.triggered.connect(self._addSumDataset)

        viewMenu = self.menuBar.addMenu("&View")
        imageMenu = viewMenu.addAction("&Image")
        imageMenu.triggered.connect(self.imageControlWindow.show)
        helpAction = self.menuBar.addAction("&Help")
        helpAction.triggered.connect(self.showHelp)

        self.menus["file"] = fileMenu
        self.menus["view"] = viewMenu
        self.menus["analysis"] = analysisMenu
        self.menus["help"] = helpAction

    def showHelp(self):
        help_window = HelpWindow(__doc__, parent=self)
        help_window.show()

    def _save_patterns_only_emc(self, fileName: Path):
        ds = self.currentDataset
        ds.patterns[ds.selectedList].write(fileName.with_suffix(".emc"))

    def _save_patterns_only_h5(self, fileName: Path):
        if isinstance(fileName, Path):
            fileName = fileName.with_suffix(".h5")
        ds = self.currentDataset
        ds.patterns[ds.selectedList].write(fileName)

    def _save_index_only_npy(self, fileName: Path):
        np.save(
            fileName.with_suffix(".npy"), self.datasetsManager.dataset1.selectedList
        )

    def _save_h5(self, fileName: Path):
        if isinstance(fileName, Path):
            fileName = ef.make_path(fileName.with_suffix(".h5"))
        self._save_patterns_only_h5(fileName)
        ef.write_array(fileName / "index", self.datasetsManager.dataset1.selectedList)

    def _save(self):
        fileTypeFuncs = {
            "(*.h5)": self._save_h5,
            "Patterns only(*.h5)": self._save_patterns_only_h5,
            "Patterns only(*.emc)": self._save_patterns_only_emc,
            "Index only(*.npy)": self._save_index_only_npy,
        }
        fileName, fileType = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save File",
            f"{self.datasetsManager.nameOfDataset1}.h5",
            ";;".join(fileTypeFuncs.keys()),
        )
        if fileName:
            fileTypeFuncs[fileType](ef.make_path(fileName))

    def _angularStatistic(self):
        if self.angularStatisticViewer is None:
            self.angularStatisticViewer = AngularStatisticViewer(
                self.datasetsManager.dataset1, bins=51
            )
            self.currentImageChanged.connect(self.angularStatisticViewer.updatePlot)
        self.angularStatisticViewer.show()
        self.currentImageChanged.emit(self)

    def _addSumDataset(self):
        currentDataset = self.datasetsManager.dataset1
        patterns = (
            currentDataset.patterns[:].sum(axis=0).reshape(1, -1)
            / currentDataset.detector.factor
            * currentDataset.detector.factor.mean()
        )
        ds = "sum:" + self.datasetsManager.nameOfDataset1
        self.datasetsManager.addDataset(
            ds, type(currentDataset)(patterns, currentDataset.detector)
        )
        # self.datasetsManager.setDataset1ByName(ds)

    def mouseMovedEvent(self, pos):
        if self._currentImage is None:
            return
        mousePoint = self.imageViewer.view.mapSceneToView(pos)
        x, y = self._transformInverted.map(mousePoint.x(), mousePoint.y())
        x_i = round(x)
        y_i = round(y)
        if (
            x_i >= 0
            and x_i < self._currentImage.shape[0]
            and y_i >= 0
            and y_i < self._currentImage.shape[1]
        ):
            v = self._currentImage[x_i, y_i]
            if np.isfinite(v):
                vstr = f"{v:.3f}"
            else:
                vstr = str(v)
            self.infoLabel.update(
                {
                    "position": f"({mousePoint.x():.2f}, {mousePoint.y():.2f})",
                    "value": vstr,
                }
            )
        else:
            self.infoLabel.update({"position": None, "value": None})

    def _setCurrentDataset(self, sl: str):
        self.updatePatternRange()

        currentDataset = self.datasetsManager.dataset1
        self._setPatternIndex(currentDataset.index)
        pidx = currentDataset.index
        self.symmetrizeCheckBox.setChecked(currentDataset.symmetrize)
        self.applyMaskCheckBox.setChecked(currentDataset.applyMask)

        if sl == "":
            self.setImage(None)
            return

        self.datasetsManager.select(pidx)

    def updatePatternRange(self):
        numData = len(self.datasetsManager.dataset1)
        self.patternIndexSpinBox.blockSignals(True)
        self.patternSlider.blockSignals(True)

        self.patternIndexSpinBox.setRange(0, numData - 1)
        self.patternSlider.setMinimum(0)
        self.patternSlider.setMaximum(numData - 1)
        self.patternNumberLabel.setText(f"/{numData}")

        self.patternIndexSpinBox.blockSignals(False)
        self.patternSlider.blockSignals(False)

    def _setPatternIndex(self, idx):
        self.patternIndexSpinBox.blockSignals(True)
        self.patternSlider.blockSignals(True)
        if idx is None:
            self.patternIndexSpinBox.setEnabled(False)
            self.patternSlider.setEnabled(False)
            self.patternIndexSpinBox.setValue(0)
        else:
            self.patternIndexSpinBox.setEnabled(True)
            self.patternSlider.setEnabled(True)
            self.patternIndexSpinBox.setValue(idx)
            self.patternSlider.setValue(idx)
        self.patternIndexSpinBox.blockSignals(False)
        self.patternSlider.blockSignals(False)

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

    def _getSetImageArgs(self):
        if self._imageInitialized:
            return dict()
        return dict(
            autoRange=False,
            autoHistogramRange=False,
            autoLevels=False,
        )

    def updateImage(self):
        self.setImage(self.datasetsManager.dataset1.getSelectedImage())

    def setImage(self, img: Optional[np.ndarray]):
        if img is None:
            self.infoLabel.update(
                info={
                    "dataset": self.datasetsManager.nameOfDataset1,
                    "index": f"-/{self.datasetsManager.dataset1.patterns.shape[0]:06d}",
                    "sum": "-",
                    "rotation": f"{self.rotationSlider.value()}°",
                }
            )
            return

        s = self.datasetsManager.dataset1.patterns[
            self.datasetsManager.dataset1.rawIndex
        ].sum()
        info = {
            "dataset": self.datasetsManager.nameOfDataset1,
            "index": f"{self.datasetsManager.dataset1.rawIndex:06d}/{self.datasetsManager.dataset1.patterns.shape[0]:06d}",
            "sum": s,
            "rotation": f"{self.rotationSlider.value()}°",
        }
        self.infoLabel.update(info)
        self._currentImage = img
        if img is None:
            self.imageViewer.clear()
            return
        sx, sy = img.shape
        y0, y1, x0, x1 = self.datasetsManager.dataset1.detectorRender.frame_extent()
        tr = QtGui.QTransform()
        tr.rotate(self.rotation)
        if self.flipCheckBox.isChecked():
            tr.scale((x0 - x1) / sx, (y1 - y0) / sy)
        else:
            tr.scale((x1 - x0) / sx, (y1 - y0) / sy)
        tr.translate(x0 - 0.5, y0 - 0.5)
        self._transform = tr
        self._transformInverted = tr.inverted()[0]
        f = self.applyImageFuncBox.text()
        if f == "":
            img = img
        else:
            with np.errstate(all="ignore"):
                img = eval(f"lambda x: {f}", {"ang_stat": ang_stat, "np": np})(img)
        self.imageViewer.setImage(img, transform=tr, **self._getSetImageArgs())
        self._imageInitialized = False
        self.currentImageChanged.emit(self)


def patternViewer(src, detector=None):
    if detector is not None:
        if isinstance(src, (ef.PatternsSOneEMC, ef.PatternsSOne)):
            pattern = src
        else:
            pattern = ef.file_patterns(src)
        det = ef.detector(detector)
        datasets = {"(default)": PatternDataModel(pattern, detector=det, modify=False)}
    else:
        datasets = {
            k: v if isinstance(v, PatternDataModelBase) else PatternDataModel(**v)
            for k, v in src.items()
        }

    return PatternViewer(datasets)
