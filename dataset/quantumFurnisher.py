import argparse
import codeConstants as cc
import os

class QuantumFurnisher:
    def __init__(self, program, language, version, simulator):
        self.language = language
        self.version = version
        self.simulator = simulator
        self.program = program
        lines = program.splitlines()
        self.lines = []
        for line in lines:
            if line != '':
                self.lines.append(line)
    
    def programRepr(self):
        text = ""
        for line in self.translatedProgram:
            text += line + "\n"
        return text


    def getLines(self):
        return self.lines
    
    def translate(self):
        if self.language == "Qiskit":
            self.translateToQiskit()
        return self.translatedProgram
    
    def isArgGate(self, gate):
        argGates = ["RZ", "CP", "CRZ"]
        return gate in argGates
    
    def qiskitGateTranslation(self, gateToBeInterpreted):
        gateInfo = gateToBeInterpreted.split('(')
        gate = gateInfo[0]
        gateArgs = gateInfo[1][:-1].split(',')
        if not self.isArgGate(gate):
            newLine = "qc." + gate.lower() +"(qr[" + gateArgs[0] + "]"
            for k in range(1, len(gateArgs)):
                newLine += ", qr[" + gateArgs[k] + "]"
            newLine += ")"
        else:
            args = gateArgs[0].split("/")
            arg = args[1]
            if args[0][0] == "-":
                newLine = "qc." + gate.lower() + "(-math.pi/" + arg + ", qr[" + gateArgs[1] + "]"
            else:
                newLine = "qc." + gate.lower() + "(math.pi/" + arg + ", qr[" + gateArgs[1] + "]"
            for k in range(2,len(gateArgs)):
                newLine += ", qr[" + gateArgs[k] + "]"
            newLine += ")"
        return newLine
    
    def qiskitMeasureTranslation(self, qbit, cbit):
        newLine = "qc.measure(qr[" + qbit + "], cr[" + cbit + "])"
        return newLine
    
    def translateToQiskit(self):
        translatedProgram = []
        insideForCounter = 0
        for line in self.lines:
            words = line.split()
            command = words[0]
            if command == "QREG":
                newLine = "qr = QuantumRegister(" + words[1] + ")"
                translatedProgram.append(newLine)
            elif command == "CREG":
                newLine = "cr = ClassicalRegister(" + words[1] + ")"
                translatedProgram.append(newLine)
                newLine = "qc = QuantumCircuit(qr,cr)"
                translatedProgram.append(newLine)
            elif command == "GATE":
                newLine = self.qiskitGateTranslation(words[1])
                for k in range(insideForCounter):
                    newLine = "\t" + newLine
                translatedProgram.append(newLine)
            elif command == "SEQGATE":
                gate = words[1]
                init = int(words[3])
                end = int(words[5])
                for k in range(init, end+1):
                    gateToBeInterpreted = gate + "(" + str(k) + ")"
                    newLine = self.qiskitGateTranslation(gateToBeInterpreted)
                    for k in range(insideForCounter):
                        newLine = "\t" + newLine
                    translatedProgram.append(newLine)
            elif command == "FOR":
                varName = words[1]
                begin = words[3]
                end = str(int(words[5]) + 1)
                newLine = "for " + varName + " in range(" +  begin + "," + end + "):"
                for k in range(insideForCounter):
                    newLine = "\t" + newLine
                translatedProgram.append(newLine)
                insideForCounter += 1
            elif command == "ENDFOR":
                insideForCounter -= 1
            elif command == "SEQMEASURE":
                begin = int(words[2])
                end = int(words[4])
                for k in range(begin, end+1):
                    newLine = self.qiskitMeasureTranslation(str(k),str(k))
                    translatedProgram.append(newLine)
            elif command == "MEASURE":
                qbit = words[1]
                cbit = words[3]
                newLine = self.qiskitMeasureTranslation(qbit, cbit)
                translatedProgram.append(newLine)
        self.translatedProgram = translatedProgram
        return translatedProgram

    
    def getPrefix(self, which):
        if which == -1:
            toBeChecked = self.language
        elif which == 0:
            toBeChecked = self.version
        else:
            toBeChecked = self.simulator
        
        if toBeChecked == "Qiskit":
            prefix = "QISKIT"
        elif toBeChecked == "CIRQ":
            prefix = "CIRQ"
        elif toBeChecked == "0.39":
            prefix = "V039"
        elif toBeChecked == "1.4.0":
            prefix = "V140"
        elif toBeChecked == "statevector":
            prefix = "STATEVECTOR"
        elif toBeChecked == "qasm":
            prefix = "QASM"
        elif toBeChecked == "basicSimulator":
            prefix = "BASICSIMULATOR"
        elif toBeChecked == "genericBackend":
            prefix = "GENERICBACKEND"
        elif toBeChecked == "aer":
            prefix = "AER"
        else:
            prefix = "NONE"
        return prefix
    
    def furnish(self):
        furnishedProgram = ""
        imports = ""
        circuit = self.programRepr()
        simulatorDecl = ""
        jobExec = ""
        simulatorOutput = ""
        circuitPrint = ""
        languagePrefix = self.getPrefix(-1)
        versionPrefix = languagePrefix + "_" + self.getPrefix(0)
        simulatorPrefix = versionPrefix + "_" + self.getPrefix(1)

        # Language-specific terms
        langImports = cc.CONST_DICT[languagePrefix + "_IMPORT"]
        imports += langImports + "\n"
        langPrint = cc.CONST_DICT[languagePrefix + "_CIRCUITPRINT"]
        circuitPrint += langPrint + "\n"

        # Version-specific terms
        if (versionPrefix + "_IMPORT") in cc.CONST_DICT:
            imports += cc.CONST_DICT[versionPrefix + "_IMPORT"] + "\n"
        verJob = cc.CONST_DICT[versionPrefix + "_JOBEXEC"] 
        jobExec += verJob + "\n"

        # Simulator-specific terms
        if (simulatorPrefix + "_IMPORT") in cc.CONST_DICT:
            imports += cc.CONST_DICT[simulatorPrefix + "_IMPORT"] + "\n"
        simulDecl = cc.CONST_DICT[simulatorPrefix + "_SIMULATORDECL"]
        simulatorDecl += simulDecl + "\n"
        simulOutput = cc.CONST_DICT[simulatorPrefix + "_SIMULATOROUT"]
        simulatorOutput += simulOutput + "\n"

        furnishedProgram = imports + "\n" + circuit + "\n" + simulatorDecl + "\n" + jobExec + "\n" + simulatorOutput + "\n"# + circuitPrint
        self.furnishedProgram = furnishedProgram
        return furnishedProgram
    
    def generateProgram(self, outFilename):
        self.translate()
        program = self.furnish()
        with open(outFilename, "w") as wfile:
            wfile.write(program)


def main():
    parser = argparse.ArgumentParser(description="Translates program from intermediate representation to chosen language and versions")
    parser.add_argument("inputDir", type=str, help="Path of input programs' directory")
    parser.add_argument("outputDir", type=str, help="Path for output directory")
    parser.add_argument("language", type=str, help="Chosen language to translate to")
    parser.add_argument("version", type=str, help="Specific version to translate to")
    parser.add_argument("simulator", type=str, help="Chosen simulator to translate to")

    args = parser.parse_args()

    files = os.listdir(args.inputDir)
    for file in files:
        filename = args.inputDir + file
        program = ""
        with open(filename) as rfile:
            program = rfile.read()
        furnisher = QuantumFurnisher(program, args.language, args.version, args.simulator)
        furnisher.getLines()
        furnisher.translate()
        program = furnisher.furnish()
        pfile = file.split('.')[0] + ".py"
        outFilename = args.outputDir + pfile
        with open(outFilename, "w") as wfile:
            wfile.write(program)


if __name__=="__main__":
    main()

    