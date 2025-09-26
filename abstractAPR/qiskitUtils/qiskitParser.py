import ast
from qiskitUtils.qiskitRepresentation import *

class qiskitParser(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.declaredRegisters = []

    def getCircuit(self):
        return self.circuit

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            varName = node.targets[0].id
            if isinstance(node.value, ast.Call):
                function = node.value.func
                if isinstance(function, ast.Name) and function.id == "QuantumRegister":
                    print(f"Quantum register declared at line {node.lineno}")
                    argument = node.value.args[0]
                    value = -1
                    if isinstance(argument, ast.Constant):
                        value = argument.value
                    self.declaredRegisters.append(QubitRegister(varName, value))
                if isinstance(function, ast.Name) and function.id == "QuantumCircuit":
                    print(f"Quantum circuit declared at line {node.lineno}")
                    self.circuit = Circuit(varName)
                    for argument in node.value.args:
                        if isinstance(argument, ast.Name):
                            qregister = self.findRegister(argument.id)
                            if qregister is not None:
                                self.circuit.addRegister(qregister)
        self.generic_visit(node)

    def visit_Expr(self, node):
        
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            
            call = node.value.func
            if isinstance(call.value, ast.Name) and call.value.id == self.circuit.name():
                if call.attr != "measure" and call.attr != "draw":
                    gateName = call.attr
                    qubits = []
                    multiGate = True
                    for argument in node.value.args:
                        if isinstance(argument, ast.Subscript):
                            registerName = argument.value.id
                            register = self.findRegister(registerName)
                            index = -1
                            if isinstance(argument.slice, ast.Constant):
                                index = argument.slice.value
                            qubit = Qubit(register, index)
                            qubits.append(qubit)
                        elif isinstance(argument, ast.Name):
                            registerName = argument.id
                            register = self.findRegister(registerName)
                            if register is not None:
                                qubits += register.qubits()
                                multiGate = False
                    if len(qubits) > 0 and multiGate == True:
                        gate = Gate(gateName, qubits, node.lineno)
                        self.circuit.addGate(gate)
                    elif multiGate == False:
                        for qubit in qubits:
                            gate = Gate(gateName, [qubit,], node.lineno)
                            self.circuit.addGate(gate)
        self.generic_visit(node)

    def findRegister(self, regName):
        for register in self.declaredRegisters:
            if register.name() == regName:
                return register
        return None