import argparse
import os
import csv
from abstractInterpretation import AbstractInterpreter

def main():
    parser = argparse.ArgumentParser(description="Applies abstract interpretation to bug datasets and collects statistics")
    parser.add_argument("buggyProgramsDir", type=str, help="Path of buggy programs directory")
    parser.add_argument("fixedProgramsDir", type=str, help="Path for fixed programs directory")
    parser.add_argument("outputPath", type=str, help="Path for output repair statistics file")

    args = parser.parse_args()

    buggyFiles = os.listdir(args.buggyProgramsDir)
    resultsList = []
    for buggyFile in buggyFiles:
        buggyFilename = args.buggyProgramsDir + buggyFile
        prefix = buggyFile.split('.')[0].split('-')[0]
        fixedFilename = args.fixedProgramsDir + prefix + ".txt"
        defaultInterpreterArgs = [buggyFilename, 2, False, False, False, False, False, False, False, False, False, 0]
        programInterpreter = AbstractInterpreter(defaultInterpreterArgs)
        result, time, repairCost = programInterpreter.interpret()
        originalCost = AbstractInterpreter.getProgramCost(fixedFilename)
        results = [buggyFile, result, time, repairCost, originalCost]
        resultsList.append(results)
    
    with open(args.outputPath, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header (optional)
        writer.writerow(["Program name", "Result" , "Time(s)", "Repair cost"])

        for resultRecord in resultsList:
            if resultRecord[1] is None:
                writer.writerow([resultRecord[0], "Invalid", "-", "-"])
            elif resultRecord[1] is True:
                cost = max(resultRecord[3]/resultRecord[4], resultRecord[4]/resultRecord[3])
                writer.writerow([resultRecord[0], "Fixed", resultRecord[2], str(cost)])
            else:
                writer.writerow([resultRecord[0], "Not Fixed", "-", "-"])



if __name__=="__main__":
    main()