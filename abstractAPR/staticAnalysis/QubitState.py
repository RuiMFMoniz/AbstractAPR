import tokenize
from io import BytesIO
import math


class QubitState:
    def __init__(self, amp1, amp2):
        self._amp1 = amp1
        self._amp2 = amp2

    def asVector(self):
        result = []
        result.append(self._amp1)
        result.append(self._amp2)
        return result
    
    def amp1(self):
        return self._amp1
    
    def amp2(self):
        return self._amp2
    
    @staticmethod
    def getAssertion(source_code, prefix="#ASSERT"):
        assertion = ""
        tokens = tokenize.tokenize(BytesIO(source_code.encode('utf-8')).readline)
        
        for toknum, tokval, _, _, _ in tokens:
            if toknum == tokenize.COMMENT and tokval.startswith(prefix):
                assertion = tokval[len(prefix):].strip()  # Remove prefix and trim

        assertion = assertion.replace("{", "")
        assertion = assertion.replace("}", "")
        assertion = assertion.replace("|", "")
        assertion = assertion.replace(">", "")
        assertions = assertion.split(',')
        
        return assertions
    
    @staticmethod
    def stateFromText(textState):
        assertionState = []
        for q in textState:
            if q == "0":
                state = QubitState(1,0)
            elif q == "1":
                state = QubitState(0,1)
            elif q == "+":
                state = QubitState(1/math.sqrt(2), 1/math.sqrt(2))
            elif q == "-":
                state = QubitState(1/math.sqrt(2), -1/math.sqrt(2))
            else:
                print("Unrecognized token, quantum state cannot be derived from char: " + q)
                return None
            assertionState.append(state)
        #assertionState.reverse()
        return assertionState

    
    def __str__(self):
        epsilon = 0.00001
        if abs(self._amp1) < epsilon:
            return "|0>"
        elif abs(self._amp2) < epsilon:
            return "|1>"
        else:
            return "(" + str(self._amp1) + "|0> + " + str(self._amp2) + "|1>)"
        