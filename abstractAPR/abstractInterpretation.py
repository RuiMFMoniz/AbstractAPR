import argparse
import time
import random
from staticAnalysis.StaticAnalysis import StaticAnalysis
from staticAnalysis.ProgramAnalyzer import ProgramAnalyzer

class AbstractInterpreter:
    def __init__(self, interpretationArgs):
        self.filepath = interpretationArgs[0]
        self.k = interpretationArgs[1]
        self.debug = interpretationArgs[2]
        self.alpha = interpretationArgs[3]
        self.timed = interpretationArgs[4]
        self.cacheSub = interpretationArgs[5]
        self.cacheSup = interpretationArgs[6]
        self.valid = interpretationArgs[7]
        self.gate = interpretationArgs[8]
        self.why = interpretationArgs[9]
        self.rank = interpretationArgs[10]
        self.ex = interpretationArgs[11]

    @staticmethod
    def getProgramCost(filename):
        programHandler = ProgramAnalyzer()
        programHandler.getAbstractProgram(filename)
        cost = programHandler.calculateCircuitCost()
        return cost

    def interpret(self):
        ogProgramHandler = ProgramAnalyzer()
        p = ogProgramHandler.getAbstractProgram(self.filepath)
        programSize = len(p.registers()[0].qubits())
        if programSize < self.k:
            p.padQubits(self.k - programSize)
        time1 = time.time()
        if 1 < self.k:
            runResult = False
            numIter = 0
            analyzer = StaticAnalysis(p, self.k, self.debug, self.alpha, self.timed, self.cacheSub,
                                self.cacheSup, self.valid, self.gate, self.why, self.ex, self.rank)
            runResult = analyzer.run()
            buggyDomains = analyzer.getBuggyDomains()
            numberOfDomains = analyzer._domain.size()
            bestProgramHandler = ogProgramHandler
            while (len(buggyDomains) > 0) and (numIter < numberOfDomains * 2):
                numIter += 1
                chosenIndex = random.choice(buggyDomains)
                chosenDomain = analyzer._domain.get(chosenIndex)
                handlers = []
                numHandlers = 50
                for k in range(numHandlers):
                    programHandler = ProgramAnalyzer()
                    programHandler.lines = bestProgramHandler.lines
                    programHandler.update()
                    programHandler.randomProgramMutation(chosenDomain)
                    handlers.append(programHandler)
                domainRepaired = False
                domainIters = 10
                while (not domainRepaired) and (domainIters > 0):
                    print("Total iterations: " + str(numIter))
                    print("Domain iterations: " + str(domainIters))
                    domainIters -= 1
                    handlerFitnessVals = []
                    for handler in handlers:
                        p = handler.analyze()
                        programSize = len(p.registers()[0].qubits())
                        if programSize < self.k:
                            p.padQubits(self.k - programSize)
                        analyzer = StaticAnalysis(p, self.k, self.debug, self.alpha, self.timed, self.cacheSub,
                                            self.cacheSup, self.valid, self.gate, self.why, self.ex, self.rank)
                        runResult = analyzer.run()
                        if runResult:
                            domainRepaired = True
                            bestProgramHandler = handler
                            break
                        else:
                            handlerFitness = analyzer.calculateFitness(chosenIndex)
                            if handlerFitness >= 500:
                                domainRepaired = True
                                bestProgramHandler = handler
                                break
                            handlerFitnessVals.append(handlerFitness)

                    if domainRepaired:
                        break
                    else:
                        sorted_indices = sorted(range(len(handlerFitnessVals)), key=lambda i: -handlerFitnessVals[i])
                        handlerFitnessVals = [handlerFitnessVals[i] for i in sorted_indices]
                        handlers = [handlers[i] for i in sorted_indices]
                        handlers = handlers[:10]
                        handlers = handlers + handlers
                        for hk in range(len(handlers)):
                            handlers[hk].randomProgramMutation(chosenDomain)
                if domainIters > 0:
                    buggyDomains = analyzer.getBuggyDomains()

            if runResult:
                print("Result: repaired")
                print("\n<-------------------------------------------->\n")
                print(bestProgramHandler.abstractRepr())
                print("\n<-------------------------------------------->\n")
                print("Repaired circuit cost: " + str(bestProgramHandler.calculateCircuitCost()))
            else:
                print("Result: not repaired")

            time2 = time.time()
            print("Total time: " + str(time2 - time1) + " seconds")
            return runResult, time2-time1, bestProgramHandler.calculateCircuitCost()
        else:
            print("\nFound k = " + str(self.k) + " and " + " n = " + str(programSize) + ". Supports only 1<k and k<=n")
            return None, None, None


def declareArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, nargs="?", default="../oreilly/rootNot.py")
    parser.add_argument("-k", type=int, nargs='?', default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--alpha", action="store_true")
    parser.add_argument("--timed", action="store_true")
    parser.add_argument("--cacheSub", action="store_true")
    parser.add_argument("--cacheSup", action="store_true")
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--why", action="store_true")
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("-ex", type=int, nargs='?', default=0)
    return parser

def getArgs(parser):
    args = parser.parse_args()
    return args


def main():
    parser = declareArgs()
    args = getArgs(parser)
    argList = [args.f, args.k, args.debug, args.alpha, args.timed, args.cacheSub, args.cacheSup, args.valid, args.gate, args.why, args.rank, args.ex]
    interpreter = AbstractInterpreter(argList)
    interpreter.interpret()


if __name__ == "__main__":
    main()