import operator
from copy import deepcopy

import cachetools
import emcfile as ef
import numpy as np
from pyqtgraph.Qt import QtCore


class PatternDataModelBase(QtCore.QObject):
    selected = QtCore.pyqtSignal(int)
    selectedListChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        patterns,
        detector,
        initIndex=0,
        selectedList=None,
        symmetrize=False,
        applyMask=False,
        modify=True,
    ):
        super().__init__()
        self._initialized = False
        self.patterns = patterns
        self.detector = detector
        self.symmetrize = False
        self._detectorRender = None
        self._cache = cachetools.LRUCache(maxsize=32)
        self._index = initIndex
        self.modify = modify
        self.selectedList = (
            np.arange(self.patterns.shape[0])
            if selectedList is None
            else np.array(selectedList, int)
        )
        self._rawIndex = (
            None if self.index is None else int(self.selectedList[self.index])
        )
        self._applyMask = applyMask
        self._protectIndex = False
        self.setSymmetrize(symmetrize)
        self._initialized = True

    def setSymmetrize(self, symmetrize):
        if self.symmetrize == symmetrize and self._initialized:
            return
        self.symmetrize = symmetrize
        self._cache.clear()
        self.select(self.index)

    def setDetector(self, detector):
        self.detector = detector
        self._detectorRender = None
        self._cache.clear()
        self.select(self.index)

    @property
    def detectorRender(self):
        raise NotImplementedError()

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
            return self.index
        self._index = None if index == -1 else index
        if self.index is not None and self.index < len(self.selectedList):
            self.selectByRawIndex(int(self.selectedList[self.index]))
        return self.index

    def selectByRawIndex(self, rawIndex):
        self._rawIndex = rawIndex
        self._protectIndex = True
        self.selected.emit(self.index)
        self._protectIndex = False
        return self.index

    def selectNext(self, d: int = 1):
        if self.index is not None:
            self.select((self.index + d) % len(self))
        return self.index

    def selectPrevious(self, d: int = 1):
        if self.index is not None:
            self.select((self.index - d) % len(self))
        return self.index

    def selectRandomly(self):
        if self.index is not None:
            self.select(np.random.choice(len(self)))
        return self.index

    def getSelectedImage(self, render=None):
        return self.getImage(self.rawIndex, render)

    def symmetrizeImage(self, img):
        raise NotImplementedError()

    @cachetools.cachedmethod(operator.attrgetter("_cache"))
    def getImage(self, index, render=None):
        if self.rawIndex is None:
            return None
        if render is None:
            render = self.detectorRender
        ans = render.render(self.symmetrizeImage(self.patterns[index]))
        if self.applyMask and hasattr(ans, "mask"):
            ans[ans.mask] = np.nan
        return ans

    def __len__(self):
        return len(self.selectedList)


class PatternDataModel(PatternDataModelBase):
    def symmetrizeImage(self, img):
        if not self.symmetrize:
            return img
        return np.concatenate([img] * 2) / 2

    @property
    def detectorRender(self):
        if (
            (self._detectorRender is not None)
            and (self._detectorRender._symmtrize == self.symmetrize)
            and (self._detectorRender._originalDetector is self.detector)
        ):
            return self._detectorRender

        if not self.symmetrize:
            self._detectorRender = ef.det_render(self.detector)
        else:
            det_sym = deepcopy(self.detector)
            det_sym.coor *= np.array([-1, -1, 1])
            self._detectorRender = ef.det_render(
                np.concatenate([self.detector, det_sym])
            )
        self._detectorRender._symmtrize = self.symmetrize
        self._detectorRender._originalDetector = self.detector
        return self._detectorRender


class _PlainDetector:
    def __init__(self, shape):
        w = (shape[1] - 1) / 2.0
        h = (shape[0] - 1) / 2.0
        self._frame_extent = (-w, w, -h, h)

    def render(self, img):
        return img

    def frame_extent(self):
        return self._frame_extent


class ImageDataModel(PatternDataModelBase):
    def __init__(self, patterns, detector=None, *args, **kargs):
        super().__init__(patterns, detector, *args, **kargs)

    def symmetrizeImage(self, img):
        if not self.symmetrize:
            return img
        return (img + img[::-1, ::-1]) / 2

    @property
    def detectorRender(self):
        if self._detectorRender is None:
            self._detectorRender = _PlainDetector(self.patterns.shape[1:])
        return self._detectorRender


class NullPatternDataModel(QtCore.QObject):
    selected = QtCore.pyqtSignal(int)
    selectedListChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.patterns = np.full((1, 1), np.nan)
        self.rawIndex = 0
        self.index = 0
        self.symmetrize = False
        self.applyMask = False

    def getSelectedImage(self):
        return None

    def __len__(self):
        return 0

    def select(self, index):
        pass
