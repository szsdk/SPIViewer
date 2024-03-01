import copy
import dataclasses

import extra_geom
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.stats import binned_statistic_2d

from .._pattern_data_model import PatternDataModelBase


class _GeomRender:
    def __init__(self, geom: extra_geom.base.DetectorGeometryBase):
        self.geom = geom
        self.render(np.zeros(self.geom.expected_data_shape))

    def frame_extent(self):
        return self._frame_extent

    def render(self, data):
        img, c = self.geom.position_modules(data.reshape(self.geom.expected_data_shape))
        w, h = img.shape
        self._frame_extent = (-c[1], -c[1] + h, -c[0], -c[0] + w)
        return img


class _GeomAngularRender:
    def __init__(self, geom: extra_geom.base.DetectorGeometryBase, bins=180):
        self.geom = geom
        self.bins = 180
        self._position2D = (
            geom.data_coords_to_positions(
                *np.mgrid[
                    : geom.expected_data_shape[0],
                    : geom.expected_data_shape[1],
                    : geom.expected_data_shape[2],
                ]
            ).reshape(-1, 3)[:, :2]
            / geom.pixel_size
        ).T
        self._positionPolar = np.array(
            [
                np.arctan2(self._position2D[1], self._position2D[0]),
                np.linalg.norm(self._position2D, axis=0),
            ]
        )
        self.render(np.zeros(self.geom.expected_data_shape))

    def frame_extent(self):
        return self._frame_extent

    def render(self, data):
        ret = binned_statistic_2d(
            self._positionPolar[0],
            self._positionPolar[1],
            data.ravel(),
            bins=self.bins,
            statistic="sum",
        )
        self._frame_extent = (
            ret.x_edge[0],
            ret.x_edge[-1],
            ret.y_edge[0],
            ret.y_edge[-1],
        )
        img = ret.statistic
        img /= img.mean(axis=0)
        return img


class GeometryDetectorPatternDataModel(PatternDataModelBase):
    # TODO: apply mask
    def symmetrizeImage(self, img):
        return img

    @property
    def detectorRender(self):
        return _GeomRender(self.detector)


@dataclasses.dataclass
class GeomOffset:
    shift: tuple
    module: int

    def apply(self, geom0):
        if self.shift == (0, 0):
            return geom0
        geom = geom0.offset(
            self.shift,
            modules=np.s_[self.module : self.module + 1],
        )
        geom.metadata = geom0.metadata
        return geom


class ModuleROI(pg.PolyLineROI):
    def __init__(self, offset, points, **args):
        self._penSelected = pg.mkPen((255, 0, 0), width=4)
        self._penNotSelected = pg.mkPen((55, 255, 0), width=1)
        super().__init__(points, closed=True, pen=self._penNotSelected, **args)
        self.isSelected = False
        self.offset = offset

    def addSegment(self, h1, h2, index=None):
        super().addSegment(h1, h2, index=index)
        h1.hide()
        h2.hide()

    def segmentClicked(self, segment, ev=None, pos=None):
        self.sigClicked.emit(self, ev)

    def setSelectState(self, state):
        self.isSelected = state
        self.setPen(self._penSelected if state else self._penNotSelected)

    def toggleSelectState(self):
        self.isSelected = not self.isSelected
        self.setSelectState(self.isSelected)
        return self.isSelected

    def setMouseHover(self, hover):
        pg.ROI.setMouseHover(self, hover)

    def setMovable(self, hover):
        pass


class GeomRefiner(QtWidgets.QMainWindow):
    def __init__(self, patternViewer):
        super().__init__(parent=patternViewer)
        self._moduleROIs = []
        self.parent().datasetsManager.dataset1Changed.connect(self._setCurrentDataset)
        self.initUI()
        self._setCurrentDataset(self.parent().datasetsManager.nameOfDataset1)

    def _removeModuleROIs(self):
        pv = self.parent()
        for k in [
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
        ]:
            pv.shortcuts.unregister(k)
        for roi in self._moduleROIs:
            pv.imageViewer.view.removeItem(roi)
        self._moduleROIs = []

    def _setCurrentDataset(self, sl: str):
        pv = self.parent()
        if not isinstance(
            pv.datasetsManager.dataset1, GeometryDetectorPatternDataModel
        ):
            self._removeModuleROIs()
            return

        det = copy.deepcopy(pv.datasetsManager.dataset1.detector)

        for mi in range(det.expected_data_shape[0]):
            roi = ModuleROI(GeomOffset((0, 0), mi), [], movable=False, rotatable=False)
            roi.isSelected = False
            pv.imageViewer.view.addItem(roi)
            for handler in roi.handles:
                handler["item"].hide()
            self._moduleROIs.append(roi)
            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
            roi.sigClicked.connect(self._clickModuleROI)

        for roi in self._moduleROIs:
            roi.show()
        pv.shortcuts.register(
            QtCore.Qt.Key.Key_Left, lambda: self.moveModules(direction="left")
        )
        pv.shortcuts.register(
            QtCore.Qt.Key.Key_Right, lambda: self.moveModules(direction="right")
        )
        pv.shortcuts.register(
            QtCore.Qt.Key.Key_Up, lambda: self.moveModules(direction="up")
        )
        pv.shortcuts.register(
            QtCore.Qt.Key.Key_Down, lambda: self.moveModules(direction="down")
        )

        self.updateModuleROIs()

    def _clickModuleROI(self, roi, ev):
        roi.toggleSelectState()
        if ev.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        for r in self._moduleROIs:
            if r is not roi:
                r.setSelectState(False)

    def _getShift(self, direction):
        d = float(self.stepInputBox.text())
        if direction == "left":
            return (0, -d)
        elif direction == "right":
            return (0, d)
        elif direction == "up":
            return (-d, 0)
        elif direction == "down":
            return (d, 0)
        raise ValueError(f"Invalid direction {direction}")

    def initUI(self):
        # Create widgets
        stepLabel = QtWidgets.QLabel("Step size:")
        self.stepInputBox = QtWidgets.QLineEdit()

        exportGeomButton = QtWidgets.QPushButton("Export geom")
        exportGeomButton.clicked.connect(self.exportGeom)

        # Set up layouts
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(stepLabel)
        input_layout.addWidget(self.stepInputBox)
        validator = QtGui.QDoubleValidator()
        self.stepInputBox.setValidator(validator)
        self.stepInputBox.setText("0.001")

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(exportGeomButton)

        self.imageViewer = pg.ImageView()
        viewer_layout = QtWidgets.QHBoxLayout()
        viewer_layout.addWidget(self.imageViewer)

        layout.addLayout(input_layout)
        layout.addLayout(button_layout)
        layout.addLayout(viewer_layout)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.setWindowTitle("geom refiner")

    def updateModuleROIs(self):
        geom = self.parent().datasetsManager.dataset1.detector
        for mi, roi in enumerate(self._moduleROIs):
            geom = roi.offset.apply(geom)
        # self.imageViewer.setImage(np.random.rand(4, 5))
        self.imageViewer.setImage(
            self.parent().datasetsManager.dataset1.getSelectedImage(
                render=_GeomAngularRender(geom)
            ),
            autoRange=False,
            autoHistogramRange=False,
            autoLevels=False,
        )
        self.parent().datasetsManager.dataset1.setDetector(geom)
        self.parent().setImage(
            self.parent().datasetsManager.dataset1.getSelectedImage()
        )
        nm, ns, nf = geom.expected_data_shape
        p2d = (
            geom.data_coords_to_positions(
                *np.meshgrid(range(nm), [0, ns - 1], [0, nf - 1], indexing="ij")
            ).reshape(nm, -1, 3)[:, [0, 1, 3, 2], 1::-1]
            / geom.pixel_size
        )
        for p, roi in zip(p2d, self._moduleROIs):
            roi.setPoints(p)
            x, y = roi.offset.shift
        return geom

    def moveModules(self, direction):
        updated = False
        for roi in self._moduleROIs:
            if not roi.isSelected:
                roi.offset.shift = (0, 0)
            else:
                d = self._getShift(direction)
                roi.offset.shift = (d[0], d[1])
                updated = True
        if updated:
            self.updateModuleROIs()

    def exportGeom(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "geom (*.geom);;All Files (*)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if fileName:
            self.parent().datasetsManager.dataset1.detector.write_crystfel_geom(
                fileName
            )
