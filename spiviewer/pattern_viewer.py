import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import emcfile as ef
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtWidgets import QInputDialog, QMessageBox

from . import utils
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
        elif text == "e":
            if pv.currentDataset.modify:
                pv.currentDataset.setSelectedList(
                    edit_with_vim(pv.currentDataset.selectedList)
                )
            else:
                QMessageBox.warning(
                    pv,
                    "Cannot modify",
                    f"The dataset {pv.currentDatasetName} cannot be modified.",
                    QMessageBox.StandardButton.Ok,
                )
        elif text == "g":
            pv.patternIndexSpinBox.selectAll()
            pv.patternIndexSpinBox.setFocus()
        elif text == "l":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectNext(d=self._gears["nextPattern"].getSpeed())
        elif text == "h":
            self.bookmarks["0"] = pv.currentDataset.rawIndex, pv.rotation
            pv.currentDataset.selectPrevious(
                d=self._gears["previousPattern"].getSpeed()
            )
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
            pv.setRotation((pv.rotation - self._gears["left"].getSpeed()) % 360)
        elif text == "=":
            pv.setRotation((pv.rotation + self._gears["right"].getSpeed()) % 360)
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
        elif text == "D":
            ret = QMessageBox.question(
                pv,
                "MessageBox",
                f"Are you sure you want to delete the dataset [{pv.currentDatasetName}]?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret == QMessageBox.StandardButton.Yes:
                pv.deleteDataset()
        elif text == "A":
            newDatasetName, ok = QInputDialog.getText(
                pv, "Text Input Dialog", "Enter your name:"
            )
            if ok:
                pv.addDataset(newDatasetName)
        elif text == "?":
            pv.showHelp()
        elif text in self._custom:
            self._custom[text]()
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
        # for i, p in enumerate(positions):
        #     self.addFreeHandle(p, item=handles[i])
        #
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
        # raise NotImplementedError()

    def saveState(self):
        return {
            "line0": self.line0.saveState(),
            "line1": self.line1.saveState(),
        }

    def setState(self, state):
        # raise NotImplementedError()
        self.line0.setState(state["line0"])
        self.line1.setState(state["line1"])

    def paint(self, p, *args):
        self.line0.paint(p)
        self.line1.paint(p)

    # def hide(self):
    #     self.line0.hide()
    #     self.line1.hide()
    #     super().hide()


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
        self.currentImageChangedFunc = None
        # This is a shortcut function which would be called whenever the image is changed.
        # It could be modified directly. Setting it to `None` avoids the calling.
        self.currentImageChanged.connect(self._callCurrentImageChangedFunc)
        self.menus = dict()

    def _callCurrentImageChangedFunc(self):
        if self.currentImageChangedFunc is not None:
            self.currentImageChangedFunc(self)

    @property
    def currentDataset(self) -> PatternDataModelBase:
        return self.datasets.get(self.currentDatasetName, nullPatternDataModel)

    @property
    def currentDatasetName(self):
        return self._currentDatasetName

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
        self.infoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.infoLabel.setStyleSheet("color:#888888;")
        self.infoLabel.move(10, 10)
        self.infoLabel.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(self.imageViewer, 1, 0, 1, 2)

        self.datasetGroup = QtWidgets.QGroupBox("Index")
        hbox = QtWidgets.QHBoxLayout()
        self.datasetGroup.setLayout(hbox)
        self.patternIndexSpinBox = QtWidgets.QSpinBox(self)
        self.patternSlider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, parent=self
        )
        self.patternNumberLabel = QtWidgets.QLabel()
        self.currentDatasetBox = QtWidgets.QComboBox(parent=self)
        self.currentDatasetBox.addItems(self.datasets.keys())
        self.currentDatasetBox.currentTextChanged.connect(self._setCurrentDataset)
        self.patternIndexSpinBox.valueChanged.connect(
            lambda v: self.currentDataset.select(v)
        )
        self.patternSlider.valueChanged.connect(lambda v: self.currentDataset.select(v))

        self.dataset2Box = QtWidgets.QComboBox(parent=self)
        self.dataset2Box.addItems(self.datasets.keys())
        self.dataset2Box.currentTextChanged.connect(self._setDataset2)
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
            lambda r: self.setImage(self.currentDataset.getSelection())
        )

        igLayout.addWidget(QtWidgets.QLabel("rotation"), 0, 0)
        igLayout.addWidget(self.rotationSlider, 0, 1, 1, 3)

        self.flipCheckBox = QtWidgets.QCheckBox("flip")
        self.flipCheckBox.stateChanged.connect(
            lambda state: self.setImage(self._currentImage)
        )
        self.symmetrizeCheckBox = QtWidgets.QCheckBox("symmetrize")
        self.symmetrizeCheckBox.stateChanged.connect(
            lambda a: self.currentDataset.setSymmetrize(
                self.symmetrizeCheckBox.isChecked()
            )
        )
        igLayout.addWidget(self.flipCheckBox, 3, 2)
        igLayout.addWidget(self.symmetrizeCheckBox, 3, 1)

        self.applyMaskCheckBox = QtWidgets.QCheckBox("apply mask")
        self.applyMaskCheckBox.stateChanged.connect(
            lambda a: self.currentDataset.setApplyMask(
                self.applyMaskCheckBox.isChecked()
            )
        )
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

        self.colormapBox = QtWidgets.QComboBox(parent=self)
        self.colormapBox.addItems(plt.colormaps())
        self.colormapBox.currentTextChanged.connect(
            lambda cm: self.imageViewer.setColorMap(pg.colormap.getFromMatplotlib(cm))
        )
        self.colormapBox.setCurrentText("magma")
        self.colormapBox.currentTextChanged.emit("magma")
        igLayout.addWidget(QtWidgets.QLabel("colormap"), 1, 0)
        igLayout.addWidget(self.colormapBox, 1, 1)

        self.applyImageFuncBox = QtWidgets.QLineEdit(parent=self.imageControlWindow)
        self.applyImageFuncBox.setPlaceholderText("np.log(x)")
        self.applyImageFuncBox.returnPressed.connect(
            lambda: self.setImage(self._currentImage)
        )
        igLayout.addWidget(QtWidgets.QLabel("apply"), 2, 0)
        igLayout.addWidget(self.applyImageFuncBox, 2, 1, 1, 3)

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
        viewMenu = self.menuBar.addMenu("&View")
        imageMenu = viewMenu.addAction("&Image")
        imageMenu.triggered.connect(self.imageControlWindow.show)
        getHelpAction = viewMenu.addAction("&Help")
        getHelpAction.triggered.connect(self.showHelp)

        self.menus["file"] = fileMenu
        self.menus["view"] = viewMenu
        self.menus["analysis"] = analysisMenu

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
        np.save(fileName.with_suffix(".npy"), self.currentDataset.selectedList)

    def _save_h5(self, fileName: Path):
        if isinstance(fileName, Path):
            fileName = ef.make_path(fileName.with_suffix(".h5"))
        self._save_patterns_only_h5(fileName)
        ef.write_array(fileName / "index", self.currentDataset.selectedList)

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
            f"{self.currentDatasetName}.h5",
            ";;".join(fileTypeFuncs.keys()),
        )
        if fileName:
            fileTypeFuncs[fileType](ef.make_path(fileName))

    def _angularStatistic(self):
        if self.angularStatisticViewer is None:
            self.angularStatisticViewer = AngularStatisticViewer(
                self.currentDataset, bins=51
            )
            self.currentImageChanged.connect(self.angularStatisticViewer.updatePlot)
        self.angularStatisticViewer.show()
        self.currentImageChanged.emit(self)

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

    def setDataset(self, name, d):
        needUpdate = name in self.datasets
        self.datasets[name] = d
        self.currentDatasetBox.addItem(name)
        self.dataset2Box.addItem(name)
        if needUpdate:
            self._setCurrentDataset(name)

    def deleteDataset(self, name=None):
        if name is None:
            name = self.currentDatasetName
        if name not in self.datasets:
            return
        index = self.currentDatasetBox.findText(name)  # find the index of text
        self.currentDatasetBox.removeItem(index)
        index = self.dataset2Box.findText(name)  # find the index of text
        self.dataset2Box.removeItem(index)
        self.datasets.pop(name)

        if len(self.datasets) == 0:
            self._setCurrentDataset("")
        elif name == self.currentDatasetName:
            self._setCurrentDataset(next(self.datasets.keys()))

    def addDataset(self, name):
        if name in self.datasets:
            raise Exception("exists")
        self.currentDatasetBox.addItem(name)
        self.dataset2Box.addItem(name)
        self.datasets[name] = PatternDataModel(
            self.currentDataset.patterns,
            detector=self.currentDataset.detector,
            initIndex=None,
            selectedList=[],
        )

    def switchDatasets(self):
        t = self._dataset2Name
        self.dataset2Box.setCurrentText(self.currentDatasetName)
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
        self.updatePatternRange()
        pidx = self.currentDataset.index
        self.symmetrizeCheckBox.setChecked(self.currentDataset.symmetrize)
        self.applyMaskCheckBox.setChecked(self.currentDataset.applyMask)

        if sl == "":
            self.setImage(None)
            return

        self.currentDataset.selected.connect(
            lambda idx: self.patternIndexSpinBox.setValue(idx)
        )
        self.currentDataset.selected.connect(
            lambda idx: self.patternSlider.setValue(idx)
        )
        self.currentDataset.selected.connect(
            lambda idx: self.setImage(self.currentDataset.getSelection())
        )
        self.currentDataset.selectedListChanged.connect(self.updatePatternRange)

        self.currentDataset.select(pidx)

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

    def _getSetImageArgs(self):
        if self._imageInitialized:
            return dict()
        return dict(
            autoRange=False,
            autoHistogramRange=False,
            autoLevels=False,
        )

    def setImage(self, img: Optional[np.ndarray]):
        s = self.currentDataset.patterns[self.currentDataset.rawIndex].sum()
        self.infoLabel.update(
            {
                "dataset": self.currentDatasetName,
                "index": f"{self.currentDataset.rawIndex:06d}/{self.currentDataset.patterns.shape[0]:06d}",
                "sum": s,
                "rotation": f"{self.rotationSlider.value()}°",
            }
        )
        self._currentImage = img
        if img is None:
            self.imageViewer.clear()
            return
        sx, sy = img.shape
        y0, y1, x0, x1 = self.currentDataset.detectorRender.frame_extent()
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
                img = eval(f"lambda x: {f}")(img)
        self.imageViewer.setImage(img, transform=tr, **self._getSetImageArgs())
        self._imageInitialized = False
        self.currentImageChanged.emit(self)


def patternViewer(src, detector=None):
    if detector is not None:
        if isinstance(src, (ef.PatternsSOneEMC, ef.PatternsSOne)):
            pattern = src
        else:
            pattern = ef.PatternsSOneEMC(src)
        det = ef.detector(detector)
        datasets = {"(default)": PatternDataModel(pattern, detector=det, modify=False)}
    else:
        datasets = {
            k: v if isinstance(v, PatternDataModelBase) else PatternDataModel(**v)
            for k, v in src.items()
        }

    return PatternViewer(datasets)
