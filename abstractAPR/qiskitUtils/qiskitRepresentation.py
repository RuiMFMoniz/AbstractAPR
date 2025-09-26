from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis


class Qubit:
    def __init__(self, register, index):
        self._register = register
        self._index = index

    def register(self):
        return self._register
    
    def index(self):
        return self._index
    
    def __str__(self):
        repr = "Qubit of index " + str(self._index) + " of register " + self._register._name
        return repr

    
    def __eq__(self, value):
        if isinstance(value, Qubit):
            return self._register._name == value._register._name and self._index == value._index
        else:
            return False

class QubitRegister:
    def __init__(self, rname, nqubits):
        self._name = rname
        self._qubits = []
        for k in range(nqubits):
            self._qubits.append(Qubit(self, k))

    def name(self):
        return self._name
    
    def qubits(self):
        return self._qubits
    
    def addQubit(self, qubit):
        self._qubits.append(qubit)
    
    def __str__(self):
        return "Register " + self._name + " holds " + str(len(self._qubits)) + " qubits\n"
    
    def __eq__(self, value):
        if isinstance(value, QubitRegister):
            return self._name == value._name
        else:
            return False
    
class Gate:
    def __init__(self, gateName, qubits, lineno):
        self._name = gateName
        self._qubits = qubits
        self._arg = []
        self._line = lineno
        self._argument = None
        if len(qubits) > 1:
            self._isMulti = True
        else:
            self._isMulti = False

    def addArgument(self, argument):
        self._argument = argument

    def argument(self):
        return self._argument
    
    @staticmethod
    def generateCxGate(index1, index2):
        name = "cx"
        reg = QubitRegister("collapsedReg", 2)
        qubit1 = Qubit(reg, index1)
        qubit2 = Qubit(reg, index2)
        gate = Gate(name, [qubit1, qubit2], -1)
        return gate
    
    @staticmethod
    def generateGateFromName(gateName, indexes):
        gateType = gateName[:-1]
        gateIndex = gateName[-1]
        reg = QubitRegister("collapsedReg", 2)
        if gateType == "cx":
            control = indexes[int(gateIndex)]
            if control == indexes[0]:
                target = indexes[1]
            else:
                target = indexes[0]
            qubit1 = Qubit(reg, control)
            qubit2 = Qubit(reg, target)
            gate = Gate(gateType, [qubit1, qubit2], -1)
        else:
            index = indexes[int(gateIndex)]
            qubit = Qubit(reg, index)
            gate = Gate(gateType, [qubit,], -1)
        return gate

    
    def name(self):
        return self._name
    
    def qubits(self):
        return self._qubits
    
    def line(self):
        return self._line
    
    def isMulti(self):
        return self._isMulti
    
    def setArgument(self, arg):
        self._arg = arg

    #def argument(self):
    #    return self._arg
    
    def incrementLine(self):
        self._line += 1

    def decrementLine(self):
        self._line -= 1
    
    def updateQubits(self, q):
        self._qubits = q
    
    def __str__(self):
        repr = ""
        if(self._isMulti):
            repr += self._name + "("
            if self.argument is not None:
                repr += str(round(self.argument, 3)) + ","
            for qubit in self._qubits:
                repr += str(qubit._index) + ","
            repr = repr[:-1]
            repr += ")"
        else: 
            repr += self._name
        repr += " --- "
        return repr
    
    def debug(self):
        repr = ""
        repr += self._name + "("
        for qubit in self._qubits:
            repr += str(qubit._index) + ","
        repr = repr[:-1]
        repr += ")"
        if len(self._arg) > 0:
            repr += " args: [ "
            for arg in self._arg:
                repr += str(arg) + " , "
        repr += "]"
        repr += " line: " + str(self._line)
        repr += " --- "
        return repr
    
    def __eq__(self, value):
        if isinstance(value, Gate):
            return self._name == value._name and self._qubits == value._qubits and self._line == value._line
        else:
            return False


class Circuit:
    def __init__(self, cname):
        self._qubitRegisters = []
        self._gates = []
        self._name = cname

    def registers(self):
        return self._qubitRegisters
    
    def gates(self):
        return self._gates
    
    def name(self):
        return self._name
    
    def addRegister(self, register):
        self._qubitRegisters.append(register)

    def addGate(self, gate):
        self._gates.append(gate)

    def removeGate(self, gate):
        self._gates.remove(gate)

    def getGates(self, qubit):
        qGates = []
        for gate in self._gates:
            if qubit in gate._qubits:
                    qGates.append(gate)
        return qGates
    
    def getQubits(self):
        nQubits = 0
        for reg in self._qubitRegisters:
            nQubits += len(reg._qubits)
        return nQubits
    
    def padQubits(self, numberOfPads):
        startIndex = len(self.registers()[0].qubits())
        for k in range(numberOfPads):
            newQubit = Qubit(self.registers()[0], startIndex)
            startIndex += 1
            self.registers()[0].addQubit(newQubit)
            
    
    def collapseCircuit(self):
        collapsed = Circuit("collapsed")
        nQubits = 0
        registerIndex = 0
        indexMap = {}
        for register in self._qubitRegisters:
            nQubits += len(register.qubits())
            indexMap[register.name()] = registerIndex
            registerIndex += len(register.qubits())
        collapsedRegister = QubitRegister("collapsedReg", nQubits)
        collapsed.addRegister(collapsedRegister)
        for gate in self._gates:
            collapsedQubits = []
            for qubit in gate.qubits():
                collapsedQubits.append(Qubit(collapsedRegister, indexMap[qubit.register().name()] + qubit.index()))
            collapsed.addGate(Gate(gate.name(), collapsedQubits, gate.line()))
        return collapsed

    def setAssertion(self, i, assertion):
        if i == 0:
            self._assertion1 = assertion
        else:
            self._assertion2 = assertion

    def getAssertion(self, i):
        if i == 0:
            return self._assertion1
        else:
            return self._assertion2
        
    def decomposeUnitaryIntoGates(self, unitary, qubitIndexes, register):
        uGate = UnitaryGate(unitary)
        qc = QuantumCircuit(2)
        qc.append(uGate, qc.qubits)

        #decomposed = qc.decompose(reps=1)

        #op = Operator(unitary)
        #decomposer = TwoQubitBasisDecomposer(CXGate())
        #decomposed = decomposer(op)

        #skd = SolovayKitaev(recursion_degree=1)
        #decomposed = skd(decomposed)
        #pass_manager = PassManager(ZXPass())
        #decomposed = pass_manager.run(decomposed)

        pass_manager = PassManager(UnitarySynthesis(basis_gates=['h', 't', 'cx', 'x', 's', 'tdg', 'sdg', 'ch', 'y', 'z', 'swap']))
        decomposed = pass_manager.run(qc)

        gateList = []
        for inst, qargs, cargs in decomposed.data:
            gateName = inst.name
            gateArgs = [decomposed.qubits.index(q) for q in qargs]
            gateParams = inst.params
            qubitList = []
            for arg in gateArgs:
                qubit = Qubit(register, qubitIndexes[arg])
                qubitList.append(qubit)
            gate = Gate(gateName, qubitList, -1)
            if len(gateParams) > 0:
                gate.setArgument(gateParams)
            gateList.append(gate)
        return gateList

    def __str__(self):
        repr = "\n<------------------------------------------------->\n"
        repr += "Circuit " + self._name + " holds " + str(len(self._qubitRegisters)) + " quantum registers.\n"
        for register in self._qubitRegisters:
            repr += str(register)
        for register in self._qubitRegisters:
            for qubit in register._qubits:
                repr += "\n   |0> --- "
                qgates = self.getGates(qubit)
                for gate in qgates:
                    repr += str(gate)
        repr += "\n\n<------------------------------------------------->\n"
        return repr