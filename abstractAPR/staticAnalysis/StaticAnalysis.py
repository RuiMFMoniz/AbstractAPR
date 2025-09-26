from functools import cmp_to_key
import numpy as np
import time
import random
from staticAnalysis.AbstractDomain import Domain
from staticAnalysis.AbstractState import AbstractState
from staticAnalysis.ComplexLibrary import ComplexMatrix
from qiskitUtils.qiskitRepresentation import Gate

class StaticAnalysis:

    whyWrong = False
    epsilonZeroMatrix = 0.0000000001
    epsilonProjection = 0.0001
    epsilonHermitian = 0.000001
    epsilonAbstractState = 0.0001
    epsilonZeroVector = 0.000001
    epsilonGramSchmidt = 0.0000000001

    def __init__(self, p, k, debug, simplealpha, timed, cacheSubsets, cacheSupersets,
                 validityCheck, showGate, whyWrongArg, extensionBase, rank):
        np.set_printoptions(formatter={'complex_kind': ComplexMatrix.complex_formatter, 'float_kind': lambda x: f"{x:.2f}"})
        self._p = p
        self._k = k
        self._debug = debug
        self._simpleAlpha = simplealpha
        self._timed = timed
        self._cacheSubsets = cacheSubsets
        self._cacheSupersets = cacheSupersets
        self._validityCheck = validityCheck
        self._showGate = showGate
        StaticAnalysis.whyWrong = whyWrongArg
        self._extensionBase = extensionBase
        self._rank = rank

        self._expansionMatrixCache = {}
        self._expansionMatrixCache["u"] = {}
        self._expansionMatrixCache["h"] = {}
        self._expansionMatrixCache["x"] = {}
        self._expansionMatrixCache["y"] = {}
        self._expansionMatrixCache["z"] = {}
        self._expansionMatrixCache["s"] = {}
        self._expansionMatrixCache["t"] = {}
        self._expansionMatrixCache["rz"] = {}
        self._expansionMatrixCache["cx"] = {}
        self._expansionMatrixCache["cz"] = {}
        self._expansionMatrixCache["ch"] = {}
        self._expansionMatrixCache["crz"] = {}
        self._expansionMatrixCache["cp"] = {}
        self._expansionMatrixCache["swap"] = {}
        self._expansionMatrixCache["ccx"] = {}
        self._expansionMatrixCache["ccz"] = {}
        self._expansionMatrixCache["cswap"] = {}

    @staticmethod
    def exp2(e):
        result = 1
        for i in range(e):
            result = 2 * result
        return result
    
    def domainPredicate(self, l):
        if (4 <= self._k) and (2 <= self._extensionBase) and (self._extensionBase <= 3):
            found = [0] * 5
            for op in self._ops:
                count = 0
                for i in op:
                    if i in l:
                        count += 1
                if count == 2:
                    found[2] = found[2] + 1
                elif count >= 3:
                    found[3] = found[3] + 1
            if self._extensionBase == 3:
                return 2 <= found[3]
            else:
                return (1 <= found[3]) or (2 <= found[2])
        else:
            return True
        
    def constantGeneration(self, i):
        indexSet = self._domain.get(i)
        size = StaticAnalysis.exp2(len(indexSet))
        m = np.zeros((size, size), dtype=complex)
        m[0][0] = 1
        return m
    
    def localProjectionGen(self, i):
        indexSet = self._domain.get(i)
        size = StaticAnalysis.exp2(len(indexSet))
        if indexSet[0] in self._unmeasuredIndexes:
            return np.identity(len(indexSet))
        col1 = self._q1[indexSet[0]].asVector()
        col2 = self._q2[indexSet[0]].asVector()
        for x in range(1, len(indexSet)):
            if indexSet[x] in self._unmeasuredIndexes:
                return np.identity(len(indexSet))
            col1 = self.tensorMultiplyColCol(self._q1[indexSet[x]], col1)
            col2 = self.tensorMultiplyColCol(self._q2[indexSet[x]], col2)
        colMatrix = np.zeros((size, size), dtype=complex)
        colMatrix[0] = col1
        colMatrix[1] = col2
        for q in range(2, size):
            colMatrix[q] = np.zeros(size, dtype=complex)
        
        return ComplexMatrix.supp(colMatrix.transpose(), self.epsilonZeroVector, self.epsilonGramSchmidt)

    def tensorMultiplyColCol(self, v1, v2):
        result = np.zeros(2 * len(v2), dtype=complex)
        for i in range(len(v2)):
            result[i] = v1.amp1() * v2[i]
            result[i+len(v2)] = v1.amp2() * v2[i]
        return result
    
    def run(self):
        ops = set()
        for gate in self._p.gates():
            qubits = gate.qubits()
            args = []
            for qubit in qubits:
                args.append(qubit.index())
            if 2 <= len(args):
                ops.add(tuple(args))

        self._ops = ops
        
        pred = self.domainPredicate
        
        if self._k == 5 and self._extensionBase == 3:
            unionsWithFive = []
            for l1 in ops:
                for l2 in ops:
                    s = set()
                    s.update(l1)
                    s.update(l2)
                    if (len(l1) == 3) and ((len(l2) == 3) and (len(s) == 5)):
                        cand = []
                        cand.extend(s)
                        cand.sort()
                        if cand not in unionsWithFive:
                            unionsWithFive.append(cand)
            dom = []
            dom.extend(unionsWithFive)
            dom.sort(key=cmp_to_key(Domain.indexSetCompare))
            programSize = len(self._p.registers()[0].qubits())
            domain = Domain(programSize, self._k, self._cacheSubsets, self._cacheSupersets)
            domain.setDomain(dom)
            #print("The domain is: " + str(domain))
        else:
            programSize = len(self._p.registers()[0].qubits())
            domain = Domain(programSize, self._k, self._cacheSubsets, self._cacheSupersets)
            domain.generateDomain(pred)

        #print("Domain size: " + str(domain.size()))

        self._domain = domain
        constantGen = self.constantGeneration
        self._abstractState = AbstractState(domain, constantGen)

        self._q1 = self._p.getAssertion(0)#[::-1]
        self._q2 = self._p.getAssertion(1)#[::-1]

        programSize = len(self._p.registers()[0].qubits())
        if (programSize < len(self._q1)) or (programSize < len(self._q2)):
            print("The assertion has wrong length!")
            return None
        
        numberOfUseless1 = programSize - len(self._q1)
        numberOfUseless2 = programSize - len(self._q2)
        self._unmeasuredIndexes = []
        if (numberOfUseless1 > 0 and numberOfUseless2 > 0 and numberOfUseless2 == numberOfUseless1):
            numberOfUnmeasured = numberOfUseless1
            indexes = list(range(programSize))
            self._unmeasuredIndexes = indexes[programSize - numberOfUnmeasured:]

        localProjectionGen = self.localProjectionGen

        self._projectedSpec = AbstractState(self._domain, localProjectionGen)

        for gate in self._p.gates():
            if self._debug:
                print("\n" + str(self._abstractState))
            
            if self._showGate:
                print(gate.debug())
            
            self.abstractStep(gate)
        
        if self._debug:
            print(str(self._abstractState))

        if not self._abstractState.valid(self.epsilonZeroMatrix, self.epsilonProjection, self.epsilonHermitian):
            print("The final abstract state is invalid!")
            return None
        
        s = self._abstractState.subsetOf(self._projectedSpec, self.whyWrong, self.epsilonAbstractState, self._unmeasuredIndexes)
        
        if self._debug:
            if s:
                print("The assertion is correct")
            else:
                print("The assertion is not fulfilled")
        
        if self._debug:
            print("Projected assertion:\n " + str(self._projectedSpec))
        
        if self._timed:
            print("Time spent in gamma:     " + str(self._timeSpentInGamma) + " seconds")
            print("Time spent in transform:     " + str(self._timeSpentInTransform) + " seconds")
            print("Time spent in alpha:     " + str(self._timeSpentInAlpha) + " seconds")

        return s
    
    def getBuggyDomains(self):
        buggyDomains = []
        for i in range(self._domain.size()):
            indexSet = self._domain.get(i)
            if any(item in self._unmeasuredIndexes for item in indexSet):
                continue
            if not ComplexMatrix.subsetCheckForProjections(self._abstractState.range()[i], self._projectedSpec.range()[i], self.whyWrong, self.epsilonAbstractState):
                buggyDomains.append(i)
        return buggyDomains

    def getWrongIndexes(self):
        candidateIndexes = set()
        for i in range(self._domain.size()):
            indexSet = self._domain.get(i)
            if any(item in self._unmeasuredIndexes for item in indexSet):
                continue
            if not ComplexMatrix.subsetCheckForProjections(self._abstractState.range()[i], self._projectedSpec.range()[i], self.whyWrong, self.epsilonAbstractState):
                candidateIndexes.update(indexSet)
        return list(candidateIndexes)

        
    def abstractStep(self, gate):
        gateQubits = gate.qubits()
        expansion = []
        for qubit in gateQubits:
            expansion.append(qubit.index())
        
        #print("Apply " + gate.name() + " on qubit set: " + str(expansion))
        if self._timed:
            time1 = time.time()
            ias1 = self.gamma(expansion, self._abstractState)
            time2 = time.time()
            self._timeSpentInGamma = time2 - time1
            ias2 = self.transform(gate, expansion, ias1)
            time3 = time.time()
            self._timeSpentInTransform = time3 - time2
            self._abstractState = self.alpha(expansion, ias2)
            time4 = time.time()
            self._timeSpentInAlpha = time4 - time3
            if self._validityCheck:
                if ias1.valid(self.epsilonZeroMatrix, self.epsilonProjection, self.epsilonHermitian):
                    print("The abstract state after gamma is valid")
                else:
                    print("The abstract state after gamma is invalid")
                if ias2.valid(self.epsilonZeroMatrix, self.epsilonProjection, self.epsilonHermitian):
                    print("The abstract state after transform is valid")
                else:
                    print("The abstract state after transform is invalid")
                if self._abstractState.valid(self.epsilonZeroMatrix, self.epsilonProjection, self.epsilonHermitian):
                    print("The abstract state after alpha is valid")
                else:
                    print("The abstract state after alpha is invalid")
            
            if self._rank:
                ranks = self._abstractState.ranks()
                for i in ranks:
                    print(str(i) + " ")
                print("Geometric mean: " + str(self.rounded(self.geometricMean(ranks))))
                print("Standard deviation: " + str(self.rounded(self.stDev(ranks))))
        else:
            self._abstractState = self.alpha(expansion, self.transform(gate, expansion, self.gamma(expansion, self._abstractState)))



    
    def expansionGen(self, i):
        #print("expansionGen: i = " + str(i) + ", for which we have domain " + str(self._inState.domain().get(i)))
        #print("expansion: " + str(self._expansion))
        indexesOfSubsets = self._inState.domain().indexesOfSubsets(i, self._expansion)
        matrices = []
        for j in range(len(indexesOfSubsets)):
            indexOfSubsetj = indexesOfSubsets[j]
            union = Domain.union(self._inState.domain().get(i), self._expansion)
            #print("union: " + str(union))
            #print("in gamma, indexes of subsets: " + str(indexesOfSubsets))
            expansionSet = self.expansionSet(self._inState.domain().get(indexOfSubsetj), union)
            #print("base: " + str(self._inState.domain().get(indexOfSubsetj)))
            #print("superset: " + str(union))
            #print("expansion set: " + str(expansionSet))
            m = ComplexMatrix.expand(self._inState.get(indexOfSubsetj), expansionSet)
            matrices.append(m)
        
        proj = ComplexMatrix.intersectProjections(matrices, self.epsilonZeroVector, self.epsilonGramSchmidt)
        return proj
    
    
    def gamma(self, expansion, inState):
        #print("gamma: start")
        self._inState = inState
        self._expansion = expansion
        newState = AbstractState(self._inState.domain(), self.expansionGen)
        return newState

    def expandAndCache(self, gateName, ex, u):
        ex = tuple(ex)
        if ex in self._expansionMatrixCache[gateName]:
            Uex = self._expansionMatrixCache[gateName][ex]
        else:
            Uex = ComplexMatrix.expand(u, ex)
            if gateName != "u" and gateName != "crz" and gateName != "rz" and gateName != "cp":
                self._expansionMatrixCache[gateName][ex] = Uex
        return Uex
    

    def transformationGen(self, i):
        #print("transformationGen, i = " + str(i))
        q = self._inState.get(i)
        descending = self.descendingChecker(self._expansion)
        if self._gate.name() == "rz" or self._gate.name() == "crz" or self._gate.name() == "cp":
            u = ComplexMatrix.string2ArgMatrix(self._gate.name(), self._gate.argument(), descending)
        else:
            u = ComplexMatrix.string2Matrix(self._gate.name(), descending)
        qubitIndexes = []
        for qubit in self._gate.qubits():
            qubitIndexes.append(qubit.index())
        qubitIndexes.sort()
        
        union =  Domain.union(self._inState.domain().get(i), self._expansion)
        #print("base: " + str(qubitIndexes))
        #print("superset: " + str(union))
        ex = self.expansionSet(qubitIndexes, union)
        #print("base: " + str(qubitIndexes))
        #print("superset: " + str(union))
        #print("expansion set: " + str(ex))
        uex = self.expandAndCache(self._gate.name(), ex, u)
        #print("Uex")
        #print(str(uex))
        result = uex @ q @ ComplexMatrix.myConjTrans(uex)
        return result


    def transform(self, gate, expansion, inState):
        #print("transform: start")
        self._expansion = expansion
        self._inState = inState
        self._gate = gate
        newState = AbstractState(inState.domain(), self.transformationGen)
        return newState
    

    def contractionGen(self, i):
        #print("contractionGen, i = " + str(i))
        domain = self._inState.domain()
        if self._simpleAlpha:
            indexesOfSupersets = []
            indexesOfSupersets.append(i)
        else:
            indexesOfSupersets = domain.indexesOfSupersets(i, self._expansion)

        #print("In alpha, indexesOfSupersets: " + str(indexesOfSupersets))
        matrices = []
        for j in range(len(indexesOfSupersets)):
            base = Domain.union(domain.get(indexesOfSupersets[j]), self._expansion)
            #print("subset: " + str(domain.get(i)))
            #print("base: " + str(base))
            cs = self.contractionSet(domain.get(i), base)
            #print("ContractionSet: " + str(cs))
            matrices.append(ComplexMatrix.supp(ComplexMatrix.traceout(self._inState.get(indexesOfSupersets[j]), cs, len(base)), self.epsilonZeroVector, self.epsilonGramSchmidt))

        return ComplexMatrix.intersectProjections(matrices, self.epsilonZeroVector, self.epsilonGramSchmidt)


    def alpha(self, expansion, inState):
        #print("alpha: start")
        self._expansion = expansion
        self._inState = inState
        #for i in range(len(self._inState.range())):
            #print("Matrix" + str(self._inState.domain().get(i)))
            #print(str(self._inState.get(i)))
        newState = AbstractState(self._inState.domain(), self.contractionGen)
        return newState
    
    def getInitialPopulation(self):
        universalGateSet = ['h0', 'h1', 't0', 't1', 'cx0', 'cx1', 'i0', 'i1']
        populationSize = 10
        patchSize = 6
        population = []
        for i in range(populationSize):
            patch = []
            for j in range(patchSize):
                patch.append(random.choice(universalGateSet))
            population.append(patch)
        return population
    
    def calculateFitness(self, domainIndex):
        fitness = 0
        parameter = 0.5   # parameter to dampen matrix difference term
        for i in range(self._domain.size()):
            indexSet = self._domain.get(i)
            if any(item in self._unmeasuredIndexes for item in indexSet):
                continue
            assertionProj = self._projectedSpec.range()[i]
            currentProj = self._abstractState.range()[i]
            if not self._abstractState.valid(self.epsilonZeroMatrix, self.epsilonProjection, self.epsilonHermitian):
                return 0
            if ComplexMatrix.subsetCheckForProjections(currentProj, assertionProj, self.whyWrong, self.epsilonAbstractState):
                if i == domainIndex:
                    fitness += 1000
                fitness += 1
            else:
                assertion = assertionProj.flatten()
                current = currentProj.flatten()
                if np.linalg.norm(current) == 0:
                    fitness -= 1
                else:
                    val = (assertion @ current) / (np.linalg.norm(current) * np.linalg.norm(assertion))
                    fitness += abs(val * parameter)
        return fitness

    
    def getFitnessOfPopulation(self, population, domainIndex):
        indexes = self._domain.get(domainIndex)
        fitnessList = []
        for patch in population:
            currentState = self._abstractState
            for gateName in patch:
                if gateName != 'i0' and gateName != 'i1':
                    gate = Gate.generateGateFromName(gateName, indexes)
                    self.abstractStep(gate)
            fitness = self.calculateFitness(domainIndex)
            fitnessList.append(fitness)
            self._abstractState = currentState
        return fitnessList
    
    def applyGate(self, gateName, index):
        indexes = self._domain.get(index)
        if gateName != 'i0' and gateName != 'i1':
            gate = Gate.generateGateFromName(gateName, indexes)
            self.abstractStep(gate)

    def compressPatch(self, gateList, index):
        currentState = self._abstractState
        compressedPatch = gateList.copy()
        for k in range(len(gateList)-1):
            trialPatch = compressedPatch
            print(gateList)
            trialPatch.remove(gateList[k])
            for gate in trialPatch:
                self.applyGate(gate, index)
            if currentState == self._abstractState:
                compressedPatch = trialPatch
            self._abstractState = currentState
        return compressedPatch

    def chooseProgenitors(self, fitnessVals):
        progNum = 4
        totalFitness = np.sum(fitnessVals)
        probabilities = [x / totalFitness for x in fitnessVals]
        indexList = range(len(fitnessVals))
        print(indexList)
        chosenProgs = []
        print(fitnessVals)
        while progNum > 0:
            choice = np.random.choice(indexList, p=probabilities)
            chosenProgs.append(choice)
            progNum -= 1
        return chosenProgs
    
    def createNewGeneration(self, progenitors, oldGeneration):
        universalGateSet = ['h0', 'h1', 't0', 't1', 'cx0', 'cx1', 'i0', 'i1']
        popSize = 10
        offpringPerParents = 4
        newPopulation = []
        for i in range(0, len(progenitors), 2):
            p1 = progenitors[i]
            p2 = progenitors[i+1]
            for n in range(offpringPerParents):
                offpring = []
                for k in range(len(p1)):
                    possible = [p1[k], p2[k]]
                    chosen = random.choice(possible)
                    offpring.append(chosen)
                mutateIndex = random.randint(0, len(offpring) - 1)
                mutateGate = random.choice(universalGateSet)
                offpring[mutateIndex] = mutateGate
                newPopulation.append(offpring)
        while len(newPopulation) < popSize:
            chosenOld = random.choice(oldGeneration)
            newPopulation.append(chosenOld)
            oldGeneration.remove(chosenOld)
        return newPopulation


    def repair(self):
        print("\nStarting repair...\n")
        repaired = False
        numIter = 0
        while (not repaired) and (numIter < 10):
            candidateIndexes = []
            for i in range(self._domain.size()):
                indexSet = self._domain.get(i)
                if any(item in self._unmeasuredIndexes for item in indexSet):
                    continue
                if not ComplexMatrix.subsetCheckForProjections(self._abstractState.range()[i], self._projectedSpec.range()[i], self.whyWrong, self.epsilonAbstractState):
                    candidateIndexes.append(i)

            if len(candidateIndexes) == 0:
                repaired = True
            else:
                numIter += 1
                chosenIndex = -1
                for i in candidateIndexes:
                    if ComplexMatrix.checkUnitaryTransformation(self._abstractState.range()[i], self._projectedSpec.range()[i]):
                        chosenIndex = i
                        break
                
                if chosenIndex != -1:
                    print("Deriving patch for projection of index " + str(chosenIndex) + " ...")
                    population = self.getInitialPopulation()
                    patchFound = False
                    patchIter = 0
                    while not patchFound and patchIter < 50:
                        patchIter += 1
                        bestScore = 0
                        bestFit = None
                        #print(population)
                        fitnessVals = self.getFitnessOfPopulation(population, chosenIndex)
                        print(fitnessVals)
                        for k in range(len(fitnessVals)):
                            if fitnessVals[k] >= 1000:
                                patchFound = True
                                if fitnessVals[k] > bestScore:
                                    bestScore = fitnessVals[k]
                                    bestFit = population[k]
                        if patchFound:
                            bestFit = self.compressPatch(bestFit, chosenIndex)
                            for gate in bestFit:
                                self.applyGate(gate, chosenIndex)
                        else:
                            reprodIndexes = self.chooseProgenitors(fitnessVals)
                            progenitors = []
                            for index in reprodIndexes:
                                progenitors.append(population[index])
                            population = self.createNewGeneration(progenitors, population)

                    print("\nPatch: ")
                    for g in bestFit:
                        #print(g.debug()) 
                        print(bestFit)
                    print("\nNew abstract state:")
                    print(str(self._abstractState))
                    print("\nAssertion Matrices:")
                    print(str(self._projectedSpec))  
                    print("\n")
                else:
                    print("No direct unitaries exist, other mechanisms should be employed...")
                    break

        if numIter >= 100:
            print("Repair was not successful, iteration limit exceeded")
        else:
            print("Repair was successful! Number of repair iterations: " + str(numIter))
    
    
    def expansionSet(self, base, superset):
        result = []
        multiplicity = 0
        basei = 0
        supersetj = 0
        while basei < len(base):
            if superset[supersetj] < base[basei]:
                multiplicity += 1
                supersetj += 1
            elif base[basei] < superset[supersetj]:
                print("expansionSet: not a superset")
                return None
            else:
                result.append(multiplicity)
                multiplicity = 0
                basei += 1
                supersetj += 1
        #if len(superset) - supersetj > 0:
        result.append(len(superset) - supersetj)
        return result
    
    
    
    def contractionSet(self, subset, base):
        result = []
        index = 0
        subseti = 0
        basej = 0
        while basej < len(base):
            if subseti < len(subset):
                if base[basej] < subset[subseti]:
                    result.append(index)
                    basej += 1
                elif subset[subseti] < base[basej]:
                    subseti += 1
                else:
                    subseti += 1
                    basej += 1
            else:
                result.append(index)
                basej += 1
            index += 1
        return result
    
    
    def geometricMean(self, values):
        logValues = []
        for v in values:
            logValues.append(np.log(v))
        
        geometricMean = self.geometricMeanFromLog(logValues)
        return geometricMean
    
    def geometricMeanFromLog(self, logValues):
        logArithmeticMean = self.arithmeticMean(logValues)
        geometricMean = np.exp(logArithmeticMean)
        return geometricMean
    
    def arithmeticMean(self, values):
        size = len(values)
        sum = self.summation(values)
        arithmeticMean = sum / size
        return arithmeticMean
    
    def summation(self, values):
        sum = 0
        for v in values:
            sum += v
        return sum
    
    def rounded(self, d):
        epsilon = 0.00001
        factor = 10000
        rounded_d = round(factor * d) / factor
        return rounded_d
    
    def stDev(self, numArray):
        sum = 0
        standardDeviation = 0
        length = len(numArray)
        for num in numArray:
            sum += num
        mean = sum/length
        for num in numArray:
            standardDeviation += np.pow(num - mean, 2)
        
        return np.sqrt(standardDeviation/length)
    
    def descendingChecker(self, expansion):
        result = True
        for i in range(len(expansion)-1):
            if expansion[i] <= expansion[i+1]:
                return False
        return result