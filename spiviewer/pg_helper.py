import numpy as np
import emcfile as ef
import pyqtgraph as pg
from pyqtgraph.exporters import Exporter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.GraphicsScene import exportDialog

translate = QtCore.QCoreApplication.translate


class ROIExporterDialog(exportDialog.ExportDialog):
    def updateItemList(self, select=None):
        super().updateItemList(select)
        sroi = QtWidgets.QTreeWidgetItem(["ROI"])
        sroi.gitem = self.scene
        self.ui.itemTree.addTopLevelItem(sroi)
        for item in self.scene.items():
            if isinstance(item, pg.ROI) and not isinstance(item.parentItem(), pg.ROI):
                si = QtWidgets.QTreeWidgetItem([str(item)])
                si.gitem = item
                sroi.addChild(si)
        sroi.setExpanded(True)


class ROIExporter(Exporter):
    Name = "ROIExporter"

    def __init__(self, item):
        Exporter.__init__(self, item)
        self.params = Parameter(
            name="params",
            type="group",
            children=[
                {
                    "name": "Name",
                    "title": translate("Exporter", "Name"),
                    "type": "str",
                    "value": "ROI",
                }
            ],
        )

    def parameters(self):
        return self.params

    def export(self, fileName=None):
        if not isinstance(self.item, pg.ROI):
            raise Exception("only support ROI export")
        if fileName is None:
            self.fileSaveDialog(filter=["*.h5", "*.hdf", "*.hd5"])
            return
        if isinstance(self.item, pg.PolyLineROI):
            ef.write_obj_h5(
                f"{fileName}::{self.params['Name']}",
                {
                    "closed": self.item.closed,
                    "points": np.array(
                        [tuple(p) for p in self.item.getState()["points"]]
                    ),
                },
                overwrite=True,
            )
        elif isinstance(self.item, pg.CircleROI):
            state = self.item.getState()
            ef.write_obj_h5(
                f"{fileName}::{self.params['Name']}",
                {
                    "pos": np.array(state["pos"]),
                    "size": np.array(state["size"]),
                    "angle": state["angle"],
                },
                overwrite=True,
            )
            print(self.item.__dict__)
        elif isinstance(self.item, pg.RectROI):
            state = self.item.getState()
            ef.write_obj_h5(
                f"{fileName}::{self.params['Name']}",
                {
                    "pos": np.array(state["pos"]),
                    "size": np.array(state["size"]),
                    "centered": state["centered"],
                },
                overwrite=True,
            )
        elif isinstance(self.item, pg.EllipseROI):
            state = self.item.getState()
            ef.write_obj_h5(
                f"{fileName}::{self.params['Name']}",
                {
                    "pos": np.array(state["pos"]),
                    "size": np.array(state["size"]),
                    "angle": state["angle"],
                },
                overwrite=True,
            )
        else:
            raise NotImplementedError(str(type(self.item)))


ROIExporter.register()
