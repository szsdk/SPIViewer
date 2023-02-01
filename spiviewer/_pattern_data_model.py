import operator

import cachetools
import emcfile as ef
import numpy as np
from pyqtgraph.Qt import QtCore


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
    selected = QtCore.pyqtSignal(int)
    selectedListChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        patterns,
        detector=None,
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
        self.detectorRender = None
        self.symmetrize = False
        self._cache = cachetools.LRUCache(maxsize=32)
        self._index = initIndex
        self.modify = modify
        self.selectedList = (
            np.arange(self.patterns.shape[0]) if selectedList is None else selectedList
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
        if self.index is not None:
            self.selectByRawIndex(int(self.selectedList[self.index]))

    def selectByRawIndex(self, rawIndex):
        self._rawIndex = rawIndex
        self._protectIndex = True
        self.selected.emit(self.index)
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
                ans = self.detectorRender.render(np.concatenate([img] * 2)) / 2
        if self.applyMask and hasattr(ans, "mask"):
            ans[ans.mask] = np.nan
        return ans

    def __len__(self):
        return len(self.selectedList)


class NullPatternDataModel(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.patterns = np.full((1, 1), np.nan)
        self.rawIndex = 0
        self.index = 0
        self.symmetrize = False
        self.applyMask = False

    def getSelection(self):
        return None

    def __len__(self):
        return 0

    def select(self, index):
        pass
