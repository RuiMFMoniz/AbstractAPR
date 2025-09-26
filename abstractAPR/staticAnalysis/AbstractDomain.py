
class Domain:

    def __init__(self, n, k, cacheSubsets, cacheSupersets):
        self._n = n
        self._k = k
        self._cacheSubsets = cacheSubsets
        self._cacheSupersets = cacheSupersets

    def generateDomain(self, pred):
        self._pred = pred
        self._domain = []
        self.addDomain([], 0, self._n, self._k, pred)
        self.initCaching()

    def setDomain(self, domain):
        self._domain = domain
        self.initCaching()
    
    def initCaching(self):
        if (self._cacheSubsets):
            self._indexesOfSubsetsCache = {}
            for i in range(len(self._domain)):
                self._indexesOfSubsetsCache[i] = {}

        if (self._cacheSupersets):
            self._indexesOfSupersetsCache = {}
            for i in range(len(self._domain)):
                self._indexesOfSupersetsCache[i] = {}     

    def addDomain(self, indexSet, low, high, count, pred):
        if (count==0):
            if pred(indexSet):
                self._domain.append(indexSet)
        else:
            for i in range(low, high):
                clonedIndexSet = indexSet.copy()
                clonedIndexSet.append(i)
                self.addDomain(clonedIndexSet, i+1, high, count-1, pred)

    def get(self, i):
        return self._domain[i]
    
    def size(self):
        return len(self._domain)
    
    @staticmethod
    def indexSetCompare(o1, o2):
        for i in range(len(o1)):
            index1 = o1[i]
            index2 = o2[i]
            if (index1 < index2):
                return -1
            elif (index1 > index2):
                return 1
        return 0
    
    @staticmethod
    def indexSetSubsetCheck(o1, o2):
        i1 = 0
        i2 = 0
        o1SubsetOfo2 = True
        while (i1 < len(o1)) and (i2 < len(o2)):
            if o1[i1] < o2[i2]:
                o1SubsetOfo2 = False
                i1 = i1 + 1
            elif o2[i2] < o1[i1]:
                i2 = i2 + 1
            else:
                i1 = i1 + 1
                i2 = i2 + 1
        if i1 < len(o1):
            o1SubsetOfo2 = False
        return o1SubsetOfo2
    
    def indexOf(self, key):
        first = 0
        last = len(self._domain) - 1
        mid = int((first + last) / 2)
        while first <= last:
            compareResult = Domain.indexSetCompare(self._domain[mid],key)
            if compareResult == -1:
                first = mid + 1
            elif compareResult == 0:
                return mid
            elif compareResult == 1:
                last = mid - 1
            mid = int((first + last) / 2)
        return -1
    
    def myAdd(self, result, candidate):
        i = self.indexOf(candidate)
        if 0 <= i:
            result.append(i)

    def indexesOfSubsets(self, i, expansion):
        if self._cacheSubsets and expansion in self._indexesOfSubsetsCache[i]:
            return self._indexesOfSubsetsCache[i][expansion]

        result = []
        current = self._domain[i]

        if len(expansion) == 1:
            x = expansion[0]
            if x in current:
                result.append(i)
            else:
                superset = self.addSingle(current, x)
                for j in range(len(superset)):
                    elem = superset[j]
                    candidate = self.removeSingle(superset, elem)
                    self.myAdd(result, candidate)
        elif len(expansion) == 2:
            x0 = expansion[0]
            x1 = expansion[1]
            b0 = x0 in current
            b1 = x1 in current
            if b0 and b1:
                result.append(i)
            elif b0 and (not b1):
                superset = self.addSingle(current, x1)
                for j in range(len(superset)):
                    elem = superset[j]
                    candidate = self.removeSingle(superset, elem)
                    self.myAdd(result, candidate)
            elif (not b0) and b1:
                superset = self.addSingle(current, x0)
                for j in range(len(superset)):
                    elem = superset[j]
                    candidate = self.removeSingle(superset, elem)
                    self.myAdd(result, candidate)
            else:
                superset = self.addSingle(self.addSingle(current, x0), x1)
                for j0 in range(len(superset)):
                    elem0 = superset[j0]
                    for j1 in range(j0+1, len(superset)):
                        elem1 = superset[j1]
                        candidate = self.removeSingle(self.removeSingle(superset, elem0), elem1)
                        self.myAdd(result, candidate)
        elif len(expansion) == 3:
            superset = self.union(self._domain[i], expansion)
            for j in range(len(self._domain)):
                if Domain.indexSetSubsetCheck(self._domain[j], superset):
                    result.append(j)
        else:
            print("Supports only 1,2,3-qubit gates")
            return None
        
        if self._cacheSubsets:
            self._indexesOfSubsetsCache[i][expansion] = result

        return result
    
    def indexesOfSupersets(self, i, expansion):
        if self._cacheSupersets and expansion in self._indexesOfSupersetsCache[i]:
            return self._indexesOfSupersetsCache[i][expansion]
        
        result = []
        subset = self._domain[i]

        if len(expansion) == 1:
            x = expansion[0]
            if x in subset:
                subsubset = self.removeSingle(subset, x)
                for j in range(self._n):
                    if j not in subsubset:
                        candidate = self.addSingle(subsubset, j)
                        self.myAdd(result, candidate)
            else:
                result.append(i)

        elif len(expansion) == 2:
            x0 = expansion[0]
            x1 = expansion[1]
            b0 = x0 in subset
            b1 = x1 in subset

            if b0 and b1:
                subsubset = self.removeSingle(self.removeSingle(subset, x0), x1)
                for j0 in range(self._n):
                    if j0 not in subsubset:
                        for j1 in range(j0+1, self._n):
                            if j1 not in subsubset:
                                candidate = self.addSingle(self.addSingle(subsubset, j0), j1)
                                self.myAdd(result, candidate)
            elif b0 and (not b1):
                subsubset = self.removeSingle(subset, x0)
                for j in range(self._n):
                    if j not in subsubset:
                        candidate = self.addSingle(subsubset, j)
                        self.myAdd(result, candidate)
            elif (not b0) and b1:
                subsubset = self.removeSingle(subset, x1)
                for j in range(self._n):
                    if j not in subsubset:
                        candidate = self.addSingle(subsubset, j)
                        self.myAdd(result, candidate) 
            else:
                result.append(i)
        elif len(expansion) == 3:
            for j in range(len(self._domain)):
                if Domain.indexSetSubsetCheck(subset, self.union(self._domain[j], expansion)):
                    result.append(j)
        else:
            print("Supports only 1,2,3-qubit gates")
            return None

        if self._cacheSupersets:
            self._indexesOfSupersetsCache[i][expansion] = result

        return result 

    @staticmethod
    def union(l1, l2):
        result = []
        for elem in l1+l2:
            if elem not in result:
                result.append(elem)
        result.sort()
        '''
        i1 = 0
        i2 = 0
        while (i1 < len(l1)) and (i2 < len(l2)):
            if l1[i1] < l2[i2]:
                result.append(l1[i1])
                i1 += 1
            elif l2[i2] < l1[i1]:
                result.append(l2[i2])
                i2 += 1
            else:
                result.append(l1[i1])
                i1 += 1
                i2 += 1
        if i1 < len(l1):
            while i1 < len(l1):
                result.append(l1[i1])
                i1 += 1
        else:
            while i2 < len(l2):
                result.append(l2[i2])
                i2 += 1
        '''
        return result
    
    def addSingle(self, l, i):
        singleton = []
        singleton.append(i)
        return self.union(l, singleton)
    
    def setDifference(self, l1, l2):
        result = []
        i1 = 0
        i2 = 0
        while (i1 < len(l1)) and (i2 < len(l2)):
            if l1[i1] < l2[i2]:
                result.append(l1[i1])
                i1 += 1
            elif l2[i2] < l1[i1]:
                i2 += 1
            else:
                i1 += 1
                i2 += 1
        if i1 < len(l1):
            while i1 < len(l1):
                result.append(l1[i1])
                i1 += 1
        else:
            while i2 < len(l2):
                i2 += 1
        
        return result
    
    def removeSingle(self, l, i):
        singleton = []
        singleton.append(i)
        return self.setDifference(l, singleton)
    
    def __str__(self):
        result = ""
        for l in self._domain:
            result += str(l) + " "
        return result


    

    


        