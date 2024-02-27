import copy
import dataclasses
import logging
from pathlib import Path

import emcfile as ef
import extra_geom
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def compute_polarization(
    polarization: str, polx: float, poly: float, norm: float
) -> float:
    """Returns polarization given pixel coordinates and type

    Parameters
    ----------
        polarization: Can be 'x', 'y' or 'none'
        polx, poly: x and y coordinates of pixel
        norm: Distance of pixel from interaction point

    Returns
    -------
        float
    """
    if polarization.lower() == "x":
        return 1.0 - (polx**2) / (norm**2)
    elif polarization.lower() == "y":
        return 1.0 - (poly**2) / (norm**2)
    elif polarization.lower() == "none":
        return 1.0 - (polx**2 + poly**2) / (2 * norm**2)
    raise Exception("Please set the polarization direction as x, y or none!")


def geom2det(
    geom: extra_geom.base.DetectorGeometryBase,
    detd: float,
    mask2,
    polarization="x",
) -> ef.Detector:
    # detd = geom.metadata["crystfel"]["clen"]
    logging.info("detector distance: %f", detd)
    pixsize = geom.pixel_size
    min_angle = np.arctan(pixsize / detd)
    qmin = 2.0 * np.sin(0.5 * min_angle)
    ewald_rad = detd / pixsize
    wavelength = 12398.419 / geom.metadata["crystfel"]["photon_energy"]
    q_sep = qmin / wavelength * (detd / ewald_rad / pixsize)

    ps = geom.get_pixel_positions().reshape(-1, 3)
    x = ps[:, 0]
    y = ps[:, 1]
    z = detd
    norm = np.sqrt(x * x + y * y + z * z)

    qscaling = 1.0 / wavelength / q_sep
    qx = x * qscaling / norm
    qy = y * qscaling / norm
    qz = qscaling * (z / norm - 1.0)

    corr = detd * (pixsize * pixsize) / np.power(norm, 3.0)
    corr *= compute_polarization(polarization, x, y, norm)
    raw_mask = np.zeros(corr.shape, dtype="u1")
    raw_mask[mask2.ravel()] = 2
    # raw_mask.reshape(16, 512, 128)[:, [0, -1], :] = 2
    # raw_mask.reshape(16, 512, 128)[:, :, [0, -1]] = 2
    corr.reshape(16, 512, 128)[:, extra_geom.agipd_asic_seams()] *= 2
    det = ef.detector(
        coor=np.array([qx, qy, qz]).T,
        factor=corr,
        mask=raw_mask,
        detd=detd / pixsize,
        ewald_rad=ewald_rad,
        norm_flag=False,
        check_consistency=False,
    )
    print(det)
    return det


@dataclasses.dataclass
class GeomOffset:
    shift: tuple
    quarter: int

    def apply(self, geom0):
        geom = geom0.offset(
            self.shift, modules=np.s_[self.quarter * 4 : (self.quarter + 1) * 4]
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
    def __init__(self, geom_fn, patternData, patternViewer):
        super().__init__(parent=patternViewer)
        pv = self.parent()
        self._patternData = patternData
        self._moduleROIs = []
        self._geom0 = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_fn)
        self._det0 = copy.deepcopy(self._patternData.detector)
        self.detd_m = self._geom0.metadata["crystfel"]["clen"]
        for mi in range(4):
            roi = ModuleROI(GeomOffset((0, 0), mi), [], movable=False, rotatable=False)
            roi.isSelected = False
            pv.imageViewer.view.addItem(roi)
            for handler in roi.handles:
                handler["item"].hide()
            self._moduleROIs.append(roi)
            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
            roi.sigClicked.connect(self._clickModuleROI)

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
        self.initUI()
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
        self.positionLabel = QtWidgets.QLabel("Step size:")
        self.positionLabel.setWordWrap(True)
        stepLabel = QtWidgets.QLabel("Step size:")
        self.stepInputBox = QtWidgets.QLineEdit()

        exportGeomButton = QtWidgets.QPushButton("Export geom")
        exportDetButton = QtWidgets.QPushButton("Export det")
        exportGeomButton.clicked.connect(self.exportGeom)
        exportDetButton.clicked.connect(self.exportDet)

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
        button_layout.addWidget(exportDetButton)

        layout.addWidget(self.positionLabel)
        layout.addLayout(input_layout)
        layout.addLayout(button_layout)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Set window properties
        self.setWindowTitle("geom refiner")

    def updateModuleROIs(self):
        geom = self._geom0
        for mi, roi in enumerate(self._moduleROIs):
            geom = roi.offset.apply(geom)
        det = geom2det(geom, self.detd_m, mask2=self._det0.mask == ef.PixelType.BAD)
        self._patternData.setDetector(det)
        self.parent().setImage(self.parent().datasetsManager.dataset1.getSelection())
        detr = ef.det_render(det)
        p2d = detr.to_cxy(det.coor)[:, ::-1].reshape(4, -1, 2)
        posStr = ""
        for p, roi in zip(p2d, self._moduleROIs):
            points = np.empty((4, 2))
            v = p[:, 0] + p[:, 1]
            points[0] = p[np.argmin(v)]
            points[2] = p[np.argmax(v)]
            v = p[:, 0] - p[:, 1]
            points[1] = p[np.argmin(v)]
            points[3] = p[np.argmax(v)]
            roi.setPoints(points)
            # size = p.max(axis=0) - pos
            x, y = roi.offset.shift
            posStr = posStr + f"Module {roi.offset.quarter}: {x:.08f}, {y:.08f}\n"

        self.positionLabel.setText(posStr)
        return geom, det

    def moveModules(self, direction):
        updated = False
        for roi in self._moduleROIs:
            if roi.isSelected:
                shift = roi.offset.shift
                d = self._getShift(direction)
                roi.offset.shift = (shift[0] + d[0], shift[1] + d[1])
                updated = True
        if updated:
            self.updateModuleROIs()

    def exportGeom(self):
        self.updateModuleROIs()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "geom (*.geom);;All Files (*)", options=options
        )
        if fileName:
            geom, _ = self.updateModuleROIs()
            geom.write_crystfel_geom(fileName)

    def exportDet(self):
        self.updateModuleROIs()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "HDF5 (*.h5);;dat (*.dat);;All Files (*)",
            options=options,
        )
        if fileName:
            _, det = self.updateModuleROIs()
            logging.info(det)
            det.write(fileName, overwrite=True)
