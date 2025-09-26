from qiskitUtils.qiskitRepresentation import *
from staticAnalysis.QubitState import QubitState
import random
import math

class ProgramAnalyzer:
    def __init__(self):
        pass

    def setProgram(self, program):
        self.program = program
        lines = program.splitlines()
        self.lines = []
        for line in lines:
            if line != '':
                self.lines.append(line)

    def __repr__(self):
        text = ""
        for line in self.lines:
            text += line + "\n"
        return text 
    
    def abstractRepr(self):
        text = ""
        for line in self.lines:
            text += line + "\n"
        return text 
    
    def getAbstractProgram(self, filename):
        program = ""
        with open(filename) as file:
            program = file.read()

        self.setProgram(program)

        lines = self.getLines()
        lines = self.extend()
        circuit = self.analyze()
        return circuit
    
    def update(self):
        self.program = self.abstractRepr()
    
    def getInstructions(self):
        init = []
        gate = []
        meas = []
        for line in self.lines:
            words = line.split()
            if words[0] == "QREG" or words[0] == "CREG":
                init.append(line)
            elif words[0] == "GATE":
                gate.append(line)
            elif words[0] == "MEASURE" or words[0] == "ASSERT":
                meas.append(line)
        return init, gate, meas
    
    def getGateOfSameType(self, gate, type):
        singleGate = ["h", "x", "z"]
        singleArgGate = ["rz"]
        doubleGate = ["swap", "cx", "cz"]
        doubleArgGate = ["crz", "cp"]
        tripleGate = ["ccx", "ccz", "cswap"]
        if type == "S":
            return random.choice(singleGate)
        elif type == "SA":
            return random.choice(singleArgGate)
        elif type == "D":
            return random.choice(doubleGate)
        elif type == "DA":
            return random.choice(doubleArgGate)
        elif type == "T":
            return random.choice(tripleGate)

    
    def getGateType(self, gateInstruction):
        singleArgGate = ["rz"]
        doubleGate = ["swap", "cx", "cz", "ch"]
        doubleArgGate = ["crz", "cp"]
        tripleGate = ["ccx", "ccz", "cswap"]
        gate = gateInstruction.split()[1].split("(")[0].lower()
        if gate in singleArgGate:
            return "SA"
        elif gate in doubleGate:
            return "D"
        elif gate in doubleArgGate:
            return "DA"
        elif gate in tripleGate:
            return "T"
        else:
            return "S"
        
    def getNumberOfQubits(self):
        quantumReg = self.lines[0]
        words = quantumReg.split()
        return int(words[1])
    
    def getLinesForQubit(self, chosenQubit):
        i, gateInstructions, m = self.getInstructions()
        lines = []
        for k in range(len(gateInstructions)):
            op = gateInstructions[k]
            indexes = op.split()[1].split('(')[1][:-1].split(',')
            for index in indexes:
                if len(index) > 3:
                    continue
                elif int(index) == chosenQubit:
                    lines.append(k)
        return lines  
    
    def randomProgramMutation(self, indexList):
        initInstructions, gateInstructions, measureInstructions = self.getInstructions()
        templates = ["CHANGE_ARG", "SWITCH_GATE", "SWITCH_QUBIT", "COPY_GATE", "MOVE_GATE", "REMOVE_GATE"]
        if len(gateInstructions) == 1:
            templates = ["CHANGE_ARG", "SWITCH_GATE", "SWITCH_QUBIT", "COPY_GATE"]

        chosenQubit = random.choice(indexList)
        candidateLines = self.getLinesForQubit(chosenQubit)
        if len(candidateLines) == 0:
            chosenLine = random.choice(range(len(gateInstructions)))
            templates = ["ADD_GATE"]
        else:
            chosenLine = random.choice(candidateLines)
        chosenGate = gateInstructions[chosenLine]
        gateType = self.getGateType(chosenGate)
        if (gateType == "S" or gateType == "D" or gateType == "T") and len(candidateLines) > 0:
            templates.remove("CHANGE_ARG")

        chosenMutation = random.choice(templates)
        if chosenMutation == "CHANGE_ARG":
            gateParams = chosenGate.split()[1].split('(')
            gateText = gateParams[0]
            gateArgs = gateParams[1][:-1].split(',')
            arg = random.choice(range(1,9))
            minus = random.randint(0,1)
            if minus == 0:
                newLine = "GATE " + gateText + "(PI/" +  str(arg)
            else:
                newLine = "GATE " + gateText + "(-PI/" +  str(arg)
            for k in range(1, len(gateArgs)):
                newLine += "," + gateArgs[k]
            newLine += ")"
            gateInstructions.insert(chosenLine,newLine)
            gateInstructions.pop(chosenLine+1)
        elif chosenMutation == "SWITCH_GATE":
            newGate = self.getGateOfSameType(chosenGate, gateType)
            gateParams = chosenGate.split()[1].split('(')
            gateArgs = gateParams[1][:-1].split(',')
            newLine = "GATE " + newGate.upper() + "(" + gateArgs[0]
            for k in range(1, len(gateArgs)):
                newLine += "," + gateArgs[k]
            newLine += ")"
            gateInstructions.insert(chosenLine,newLine)
            gateInstructions.pop(chosenLine+1)
        elif chosenMutation == "SWITCH_QUBIT":
            nQubits = self.getNumberOfQubits()
            if gateType == "S" or gateType == "SA":
                nArg = 1
            elif gateType == "D" or gateType == "DA":
                nArg = 2
            else:
                nArg = 3
            if len(indexList) < nArg:
                newQubitVals = random.sample(range(nQubits), nArg)
            else:
                newQubitVals = random.sample(indexList, nArg)
            gateParams = chosenGate.split()[1].split('(')
            gateText = gateParams[0]
            gateArgs = gateParams[1][:-1].split(',')
            if gateType == "SA" or gateType == "DA":
                newLine = "GATE " + gateText + "(" + gateArgs[0]
                for k in range(len(newQubitVals)):
                    newLine += "," + str(newQubitVals[k])
            else:
                newLine = "GATE " + gateText + "(" + str(newQubitVals[0])
                for k in range(1, len(newQubitVals)):
                    newLine += "," + str(newQubitVals[k])
            newLine += ")"
            gateInstructions.insert(chosenLine,newLine)
            gateInstructions.pop(chosenLine+1)
        elif chosenMutation == "COPY_GATE":
            chosenCopy = gateInstructions[random.choice(range(len(gateInstructions)))]
            gateInstructions.insert(chosenLine, chosenCopy)
        elif chosenMutation == "MOVE_GATE":
            gateInstructions.pop(chosenLine)
            lineToMove = random.choice(range(len(gateInstructions)))
            gateInstructions.insert(lineToMove, chosenGate)
        elif chosenMutation == "REMOVE_GATE":
            gateInstructions.pop(chosenLine)
        elif chosenMutation == "ADD_GATE":
            choice = random.randint(0,1)
            if choice == 0:
                newInstruction = "GATE H(" + str(chosenQubit) + ")"
            else:
                newInstruction = "GATE CX(" + str(indexList[0]) + "," + str(indexList[1]) + ")"
            gateInstructions.insert(chosenLine, newInstruction)
        else:
            print("UNRECOGNIZED MUTATION OPERATOR!")

        self.lines = initInstructions + gateInstructions + measureInstructions
        self.update()
        return chosenMutation 
    
    def calculateCircuitCost(self):
        mList = ["CX", "SWAP", "CZ", "CCZ", "CSWAP", "CP", "CRZ"]
        tList = ["T", "CCX"]
        sCount = 0
        mCount = 0
        tCount = 0
        nQubits = 0
        depth = 0
        lastLayer = None
        for line in self.lines:
            line = line.split()
            if line[0] == "QREG":
                nQubits = int(line[1])
                lastLayer = [0] * nQubits
            if line[0] != "GATE":
                continue
            gate = line[1]
            gateName = gate.split('(')[0]
            gateArgs = gate.split('(')[1][:-1].split(',')
            newGateArgs = []
            for g in gateArgs:
                if "PI" not in g:
                    newGateArgs.append(g)
            gateArgs = newGateArgs
            if gateName in mList:
                mCount += 1
            elif gateName in tList:
                tCount += 1
            else:
                sCount += 1

            print(lastLayer)
            print(gateArgs)
            gateLayer = max([lastLayer[int(q)] for q in gateArgs], default=0) + 1
            for q in gateArgs:
                lastLayer[int(q)] = gateLayer
            if gateLayer > depth:
                depth = gateLayer
        
        #print("Number of Qubits: " + str(nQubits))
        #print("Circuit depth: " + str(depth))
        #print("Number of Single gates: " + str(sCount))
        #print("Number of Multi gates: " + str(mCount))
        #print("Number of Non-Clifford gates: " + str(tCount))
        cost = nQubits + depth + 0.25 * sCount + mCount + 5 * tCount
        return cost


    def extend(self):
        newLines = []
        insideFor = False
        for i in range(len(self.lines)):
            line = self.lines[i]
            words = line.split()
            command = words[0]
            if command == "SEQGATE":
                gate = words[1]
                init = int(words[3])
                end = int(words[5])
                for k in range(init, end+1):
                    newCommand = "GATE " + gate + "(" + str(k) + ")"
                    newLines.append(newCommand)
            elif command == "SEQMEASURE":
                init = int(words[2])
                end = int(words[4])
                for k in range(init, end+1):
                    newCommand = "MEASURE " + str(k) + " IN " + str(k)
                    newLines.append(newCommand)
            elif command == "FOR":
                insideFor = True
                followingLines = self.lines[i+1:]
                forLoop = []
                for j in range(len(followingLines)):
                    fline = followingLines[j]
                    fwords = fline.split()
                    forCommand = fwords[0]
                    if forCommand == "ENDFOR":
                        break
                    else:
                        forLoop.append(fline)
                varName = words[1]
                beginFor = int(words[3])
                endFor = int(words[5])
                for l in range(beginFor, endFor+1):
                    for elem in forLoop:
                        addedGate = elem.replace(varName, str(l))
                        newLines.append(addedGate)

            elif command == "ENDFOR":
                insideFor = False
            else:
                if not insideFor:
                    newLines.append(line)
        self.lines = newLines
        return newLines
    

    def isSingularGate(self, gate):
        singularGates = ["H", "T", "X", "Z"]
        return gate in singularGates

    def checkNextInSequence(self, gate, arg, gateSeq, currentSeq):
        return gate == gateSeq and arg == currentSeq + 1
    
    def breakSequence(self, gateSequence, begin, end, lines):
        newLines = lines
        if end - begin > 1:
            seqLine = "SEQGATE " + gateSequence + " FROM " + str(begin) + " TO " + str(end)
            newLines.append(seqLine)
        else:
            for i in range(begin, end+1):
                seqLine = "GATE " + gateSequence + "(" + str(i) + ")"
                newLines.append(seqLine)
        return newLines

    
    def compress(self):
        newLines = []
        onSequence = False
        beginSequence = -1
        currentSequence = -1
        gateSequence = ""
        for line in self.lines:
            words = line.split()
            command = words[0]
            if command == "GATE":
                gate = words[1]
                words = gate.split('(')
                gate = words[0]
                if self.isSingularGate(gate):
                    arg = int(words[1][:-1])
                    if not onSequence:
                        onSequence = True
                        beginSequence = arg
                        currentSequence = arg
                        gateSequence = gate
                    else:
                        isNext = self.checkNextInSequence(gate, arg, gateSequence, currentSequence)
                        if isNext:
                            currentSequence = arg
                        else:
                            newLines = self.breakSequence(gateSequence, beginSequence, currentSequence, newLines)
                            onSequence = True
                            beginSequence = arg
                            currentSequence = arg
                            gateSequence = gate
                else:
                    if onSequence:
                        newLines = self.breakSequence(gateSequence, beginSequence, currentSequence, newLines)
                        onSequence = False
                    newLines.append(line)
            else:
                if onSequence:
                    newLines = self.breakSequence(gateSequence, beginSequence, currentSequence, newLines)
                    onSequence = False
                newLines.append(line)
        self.lines = newLines
        return newLines
                

    def getLines(self):
        return self.lines
    
    def minIndex(self):
        for k in range(len(self.lines)):
            line = self.lines[k]
            words = line.split()
            command = words[0]
            if command == "GATE":
                return k        

    def maxIndex(self):
        for k in range(len(self.lines)):
            line = self.lines[k]
            words = line.split()
            command = words[0]
            if command == "MEASURE":
                return k
            
    def getQubits(self):
        regLine = self.lines[0]
        words = regLine.split()
        return int(words[1])
    
    def getQubitsOfGate(self, gate):
        if gate in ["H", "X", "Z", "Y", "S", "T"]:
            return 1
        elif gate in ["CX", "CZ", "SWAP"]:
            return 2
        else:
            return 3
        
    def getMultipleGateIndexes(self):
        indexes = []
        for k in range(len(self.lines)):
            line = self.lines[k]
            words = line.split()
            command = words[0]
            if command == "GATE":
                gate = words[1].split('(')[0]
                numQubits = self.getQubitsOfGate(gate)
                if numQubits > 1:
                    indexes.append(k)
        return indexes
    
    def getArgsFromGate(self, gate):
        words = gate.split()
        gateInfo = words[1].split('(')
        gateType = gateInfo[0]
        gateArgs = gateInfo[1][:-1].split(',')
        return gateArgs
            
    def pickGate(self, qubitNumber):
        possibleGates = []
        if qubitNumber > 0:
            oneGates = ["H", "X", "Z", "Y", "S", "T"]
            possibleGates.extend(oneGates)
        if qubitNumber > 1:
            twoGates = ["CX", "CZ", "SWAP"]
            possibleGates.extend(twoGates)
        if qubitNumber > 2:
            threeGates = ["CSWAP", "CCX", "CCZ"]
            possibleGates.extend(threeGates)
        return random.choice(possibleGates)

    def mutate(self, chosenOp = None, chosenNumber = None):
        if chosenNumber is None:
            chosenNumber = random.randint(1,3)
        for k in range(chosenNumber):
            if chosenOp is None:
                operations = ["A", "D", "S", "R"]
                chosenOp = random.choice(operations)
            minIndex = self.minIndex()
            maxIndex = self.maxIndex()
            numberOfQubits = self.getQubits()
            candidates = self.lines[minIndex:maxIndex]
            chosenIndex = random.randint(minIndex, maxIndex)
            if chosenOp == "A":
                gate = self.pickGate(numberOfQubits)
                nQubits = self.getQubitsOfGate(gate)
                qubits = range(numberOfQubits)
                args = random.sample(qubits, nQubits)
                addedGate = "GATE " + gate + "(" + str(args[0])
                for k in range(1,len(args)):
                    addedGate += "," + str(args[k])
                addedGate += ")"
                self.lines.insert(chosenIndex, addedGate)
            elif chosenOp == "D":
                if len(candidates) > 1:
                    self.lines.pop(chosenIndex)
            elif chosenOp == "R":
                gate = self.pickGate(numberOfQubits)
                nQubits = self.getQubitsOfGate(gate)
                qubits = range(numberOfQubits)
                args = random.sample(qubits, nQubits)
                addedGate = "GATE " + gate + "(" + str(args[0])
                for k in range(1,len(args)):
                    addedGate += "," + str(args[k])
                addedGate += ")"
                self.lines.insert(chosenIndex, addedGate)
                self.lines.pop(chosenIndex+1)
            elif chosenOp == "S":
                multipleQubitCandidates = self.getMultipleGateIndexes()
                chosenIndex = random.choice(multipleQubitCandidates)
                gate = self.lines[chosenIndex]
                oldArgs = self.getArgsFromGate(gate)
                qubits = range(numberOfQubits)
                newArgs = oldArgs
                while newArgs == oldArgs:
                    newArgs = random.sample(qubits, len(oldArgs))
                gateType = gate.split()[1].split('(')[0]
                addedGate = "GATE " + gateType + "(" + str(newArgs[0])
                for k in range(1, len(newArgs)):
                    addedGate += "," + str(newArgs[k])
                addedGate += ")"
                self.lines.insert(chosenIndex, addedGate)
                self.lines.pop(chosenIndex+1)


    def isArgGate(self, gate):
        if gate in ["RZ", "CP", "CRZ"]:
            return True
        else:
            return False

    def analyze(self):
        circuit = Circuit("collapsedReg")
        reg = None
        for line in self.lines:
            words = line.split()
            command = words[0]
            if command == "QREG":
                arg = int(words[1])
                register = QubitRegister("qreg", arg)
                reg = register
                circuit.addRegister(register)
            elif command == "CREG" or command == "MEASURE":
                # classical registers and measurements do not need to be modeled under abstraction
                continue
            elif command == "GATE":
                gate = words[1]
                words = gate.split('(')
                gate = words[0]
                args = words[1][:-1].split(',')
                if self.isArgGate(gate):
                    argument = math.pi / int(args[0].split("/")[1])
                    args = args[1:]
                args = list(map(int, args))
                qubits = []
                for arg in args:
                    qubit = Qubit(reg, arg)
                    qubits.append(qubit)
                newGate = Gate(gate.lower(), qubits, -1)
                if self.isArgGate(gate):
                    newGate.addArgument(argument)
                circuit.addGate(newGate)
            elif command == "ASSERT":
                assertions = words[1].split('(')[1][:-1]
                assertions = assertions.replace("|", "")
                assertions = assertions.replace("<", "")
                assertions = assertions.replace(">", "")
                assertions = assertions.split(",")
                assertionStates = []
                for a in assertions:
                    state = QubitState.stateFromText(a)
                    assertionStates.append(state)
                circuit.setAssertion(0, assertionStates[0])
                circuit.setAssertion(1, assertionStates[1])
            else:
                print("COMMAND NAME NOT RECOGNIZED: " + command)
        return circuit
    