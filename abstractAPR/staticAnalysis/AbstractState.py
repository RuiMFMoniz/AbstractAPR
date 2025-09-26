from staticAnalysis.ComplexLibrary import ComplexMatrix
import numpy as np

class AbstractState:

    def __init__(self, domain, initValGen):
        self._domain = domain
        self._range = []
        for i in range(domain.size()):
            self._range.append(initValGen(i))
    
    def get(self, i):
        return self._range[i]
    
    def domain(self):
        return self._domain
    
    def range(self):
        return self._range
    
    def subsetOf(self, other, whyWrong, epsilon, unmeasuredIndexes):
        for i in range(self._domain.size()):
            domainIndexes = self._domain.get(i)
            skip = False
            for domainIndex in domainIndexes:
                if domainIndex in unmeasuredIndexes:
                    skip = True
                    break
            if skip:
                continue
            if not ComplexMatrix.subsetCheckForProjections(self._range[i], other.range()[i], whyWrong, epsilon):
                return False
        return True
    
    def valid(self, epsilonZ, epsilonP, epsilonH):
        result = True

        minNorm1 = 1
        for i in range(self._domain.size()):
            current = ComplexMatrix.norm1(self._range[i])
            if current < minNorm1:
                minNorm1 = current
                if current < epsilonZ:
                    result = False
                    print("The matrix for " + str(self._domain.get(i)) + " is close to 0:")
                    if self._range[i].shape[0] <= 8:
                        print("The matrix is\n" + str(self._range[i]))
                    else:
                        print("The matrix has dimension " + str(self._range[i].shape[0]) + " x " + str(self._range[i].shape[0]))
        
        for i in range(self._domain.size()):
            current = self._range[i]
            if ComplexMatrix.norm1(ComplexMatrix.myMatrixSubtract(ComplexMatrix.myMatrixMul(current, current), current)) > epsilonP:
                result = False
                print("The matrix for " + str(self._domain.get(i)) + " is not a projection:")
                if self._range[i].shape[0] <= 8:
                    print("The matrix is\n" + str(current) + "\nindeed, its product with itself is:\n" + str(ComplexMatrix.myMatrixMul(current, current)))
                else:
                    print("The matrix has dimension " + str(self._range[i].shape[0]) + " x " + str(self._range[i].shape[0]))

        for i in range(self._domain.size()):
            current = self._range[i]
            if ComplexMatrix.norm1(ComplexMatrix.myMatrixSubtract(ComplexMatrix.myConjTrans(current), current)) > epsilonH:
               print("The matrix for " + str(self._domain.get(i)) + " is not Hermetian:\n" + str(current))
               result = False

        identity = np.eye(self._range[0].shape[0])
        for i in range(self._domain.size()):
            current = self._range[i]
            if np.allclose(identity, current):
                print("The matrix for " + str(self._domain.get(i)) + " is equal to Identity")
                result = False

        return result 
    
    def ranks(self):
        result = []
        for i in range(self._domain.size()):
            result.append(ComplexMatrix.rank(self._range[i]))
        return result
    
    def isIdentity(self):
        identity = np.eye(self._range[0].shape[0])
        for i in range(self._domain.size()):
            current = self._range[i]
            if np.allclose(identity, current):
                print("New state is equivalent to identity")
                return True
        return False
    
    def __str__(self):
        result = ""
        for i in range(self._domain.size()):
            result += "\nMatrix" + str(self._domain.get(i)) + "\n" + str(self._range[i])
        return result

