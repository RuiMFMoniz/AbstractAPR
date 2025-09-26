import numpy as np
import itertools


class ComplexMatrix:

    @staticmethod
    def complex_formatter(x):
        if np.iscomplexobj(x):  # Check if the value is complex
            real, imag = x.real, x.imag
            if np.isclose(imag, 0):  # If the imaginary part is close to zero, show only real
                return f"{real:.2f}"
            elif np.isclose(real, 0):  # If the real part is close to zero, show only imaginary
                return f"{imag:.2f}j"
            else:
                return f"{real:.2f} + {imag:.2f}j"
        return f"{x:.1f}"  # Format regular numbers


    @staticmethod
    def checkUnitaryTransformation(m1, m2):
        eig1 = np.linalg.eigvalsh(m1)
        eig2 = np.linalg.eigvalsh(m2)
        print("Eigenvalues of m1: " + str(eig1))
        print("Eigenvalues of m2: " + str(eig2))
        same_spectrum = np.allclose(np.sort(eig1), np.sort(eig2))
        return same_spectrum
    
    @staticmethod
    def getDegenerateIndexes(eigvals):
        indexMap = {}
        for k in range(len(eigvals)):
            if eigvals[k] not in indexMap:
                indexMap[eigvals[k]] = []
                indexMap[eigvals[k]].append(k)
            else:
                indexMap[eigvals[k]].append(k)
        return indexMap
                

    @staticmethod
    def deriveUnitaryTransformation(m1, m2):
        #print("M1: ")
        #print(m1)
        #print("M2: ")
        #print(m2)

        eigvals1, eigvecs1 = np.linalg.eigh(m1)
        eigvals2, eigvecs2 = np.linalg.eigh(m2)
        #print("Eigenvalues of m1: " + str(eigvals1))
        #print("Eigenvalues of m2: " + str(eigvals2))
        #print("Eigenvectors of m1: " + str(eigvecs1))
        #print("Eigenvectors of m2: " + str(eigvecs2))

        # Sort indices
        idx1 = np.argsort(eigvals1)
        idx2 = np.argsort(eigvals2)

        # Reorder eigenvectors
        eigvecs1 = eigvecs1[:, idx1]
        eigvecs2 = eigvecs2[:, idx2]

        degenPermutations = list(itertools.permutations(range(4)))
        possibleTransformatios = []
        for permutation in degenPermutations:
            permVecs1 = eigvecs1[:,permutation]
            for p2 in degenPermutations:
                permVecs2 = eigvecs2[:,p2]
                uPerm = permVecs2 @ permVecs1.conj().T
                check = uPerm @ m1 @ uPerm.conj().T 
                if np.allclose(check, m2):
                    if not any(np.array_equal(uPerm, arr) for arr in possibleTransformatios):
                        possibleTransformatios.append(uPerm)
                        #print(check)

        return possibleTransformatios
        

    @staticmethod
    def myMatrixSubtract(m1, m2):
        diff = m1 - m2
        return diff
    
    @staticmethod
    def myMatrixMul(m1, m2):
        product = m1 @ m2
        return product
    
    @staticmethod
    def norm1(a):
        norm = 0
        ni = len(a)
        nj = len(a[0])
        for j in range(nj):
            colSum = 0
            for i in range(ni):
                colSum += np.absolute(a[i][j])
            norm = np.max([norm, colSum])
        return norm

    @staticmethod
    def myConjTrans(m):
        return np.transpose(np.conjugate(m))
    
    @staticmethod
    def rank(m):
        return np.linalg.matrix_rank(m)
    
    @staticmethod
    def doublenorm(v):
        return np.sqrt(v@v)
    
    @staticmethod
    def proj(u, v, epsilon):
        if ComplexMatrix.doublenorm(u) < epsilon:
            return np.zeros(len(u), dtype=complex)
        else:
            n = u @ v
            d = u @ u
            f = n / d
            result = np.zeros(len(u), dtype=complex)
            for i in range(len(u)):
                result[i] = u[i] * f
            return result
    
    @staticmethod
    def supp(m, epsilonZ, epsilonG):
        v = np.transpose(m)
        u = v.copy()
        for i in range(len(v)):
            for j in range(i):
                u[i] = u[i] - ComplexMatrix.proj(u[j], v[i], epsilonZ)

        e = np.zeros((len(v[0]), len(v[0])), dtype=complex)
        j = 0
        for i in range(len(u)):
            n = ComplexMatrix.doublenorm(u[i])
            if n > epsilonG:
                e[j] = u[i] / n
                j += 1

        erows = np.transpose(e)
        erowsDagger = np.transpose(np.conjugate(erows))
        return erows @ erowsDagger
    
    @staticmethod
    def traceout(u, indices, baseSize):
        result = u.copy()
        for i in range(len(indices)):
            result = ComplexMatrix.traceoutAt(result,(baseSize - 1) - indices[i])
        return result
    
    @staticmethod
    def traceoutAt(mabc, t):
        if t == 0:
            half = int(len(mabc)/ 2)
            result = np.zeros((half, half), dtype=complex)
            for i in range(half):
                for j in range(half):
                    result[i][j] = mabc[i][j] + mabc[i+half][j+half]
            return result
        elif t > 0:
            result = mabc.copy()
            result = ComplexMatrix.movekto0(result, t)
            return ComplexMatrix.traceoutAt(result, 0)
        else:
            print("Problem in traceout!")
            return None
    

    @staticmethod
    def getUMatrix(gateArgs):
        theta = gateArgs[0]
        phi = gateArgs[1]
        lamb = gateArgs[2]
        u = np.array([[np.cos(theta/2), -np.exp(1j*lamb) * np.sin(theta/2)],
                      [np.exp(1j*phi) * np.sin(theta/2), np.exp(1j * (phi + lamb)) * np.cos(theta/2)]])
        return u
    
    @staticmethod
    def string2ArgMatrix(name, argument, descending):
        rz = np.array([[np.exp(-1j * argument / 2), 0],
                       [0, np.exp(1j * argument / 2)]])
    
        invcrz = np.array([[1, 0, 0, 0],
                       [0, np.exp(-1j * argument / 2), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, np.exp(1j * argument / 2)]])

        crz = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, np.exp(-1j * argument / 2), 0],
                       [0, 0, 0, np.exp(1j * argument / 2)]])
        
        cp = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, np.exp(1j * argument)]])
        
        if name == "rz":
            return rz
        elif name == "cp":
            return cp
        elif name == "crz":
            if descending:
                return crz
            else:
                return invcrz
        else:
            print("Gate not recognized!")
            return None

    
    @staticmethod
    def string2Matrix(name, descending):
        h = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                      [1/np.sqrt(2), -1/np.sqrt(2)]]) 
        x = np.array([[0, 1],
                      [1, 0]]) 
        z = np.array([[1, 0],
                      [0, -1]]) 
        s = np.array([[1, 0],
                      [0, 1j]]) 
        t = np.array([[1, 0],
                      [0, np.exp(1j * np.pi / 4)]])   
        y = np.array([[0, -1j],
                      [1j, 0]])  
        cx = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
        invcx = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0]]) 
        cz = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, -1]]) 
        ch = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
                       [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]])
        invch = np.array([[1, 0, 0, 0],
                          [0, 1/np.sqrt(2), 0, 1/np.sqrt(2)],
                          [0, 0, 1, 0],
                          [0, 1/np.sqrt(2), 0, -1/np.sqrt(2)]])
        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        cswap = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])
        invcswap = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])
        ccx = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0]])
        invccx = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0]])
        ccz = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1]])
        
        if name == "h":
            return h
        elif name == "t":
            return t
        elif name == "s":
            return s
        elif name == "x":
            return x
        elif name == "z":
            return z
        elif name == "y":
            return y
        elif name == "cx":
            if descending:
                return cx
            else:
                return invcx
        elif name == "cz":
            return cz
        elif name == "ch":
            if descending:
                return ch
            else:
                return invch
        elif name == "swap":
            return swap
        elif name == "ccx":
            if descending:
                return ccx
            else:
                return invccx
        elif name == "cswap":
            if descending:
                return cswap
            else:
                return invcswap
        if name == "ccz":
            return ccz
        else:
            print("Gate not recognized!")
            return None

    
    @staticmethod
    def exchange2inPlace(u, x, y):
        m = u.copy()
        row = m[x].copy()
        m[x] = m[y]
        m[y] = row

        for i in range(len(m)):
            c = m[i][x].copy()
            m[i][x] = m[i][y]
            m[i][y] = c

        return m


    @staticmethod
    def swap0andk(u, k):
        m = u.copy()
        if 0 < k and np.pow(2, k) < len(m):
            numberOfQubits = int(np.log(len(m)) / np.log(2))
            half = int(len(m) / 2)
            stride = int(half / np.pow(2, k))
            steps = int(half / stride)
            for i in range(steps):
                for j in range(stride):
                    if i % 2 == 1:
                        pos = i*stride + j
                        otherPos = pos + half - int(np.pow(2, numberOfQubits - k - 1))
                        m = ComplexMatrix.exchange2inPlace(m, pos, otherPos)
        return m


    @staticmethod
    def swapkandkplus1(u, k):
        m = u.copy()
        if k == 0:
            m = ComplexMatrix.swap0andk(m, 1)
        else:
            m = ComplexMatrix.swap0andk(m, k)
            m = ComplexMatrix.swap0andk(m, k+1)
            m = ComplexMatrix.swap0andk(m, k)
        return m
    
    
    @staticmethod
    def move0tok(u, k):
        m = u.copy()
        for i in range(k):
            m = ComplexMatrix.swapkandkplus1(m, i)
        return m
    
    @staticmethod
    def movekto0(u, k):
        m = u.copy()
        if 0 < k:
            i = k - 1
            while 0 <= i:
                m = ComplexMatrix.swapkandkplus1(m, i)
                i -= 1
        return m
    
    @staticmethod
    def exp2(e):
        result = 1
        for i in range(e):
            result = 2 * result
        return result
    
    @staticmethod
    def expandAt(u, k):
        exp2k = ComplexMatrix.exp2(k)
        if exp2k == 0:
            result = np.kron(np.identity(2), u)
        elif exp2k < len(u):
            result = np.kron(np.identity(2), u)
            result = ComplexMatrix.move0tok(result, k)
        else:
            result = np.kron(u, np.identity(2))
        return result

    @staticmethod
    def expand(u, indices):
        result = u.copy()
        #indices = tuple(reversed(indices))
        for i in range(len(indices)):
            for x in range(indices[i]):
                result = ComplexMatrix.expandAt(result, (len(indices) - 1) - i)
        return result

    @staticmethod
    def intersectProjections(l, epsilonZ, epsilonG):
        np.set_printoptions(formatter={'complex_kind': ComplexMatrix.complex_formatter, 'float_kind': lambda x: f"{x:.2f}"})
        #print("At intersect projections: ")
        n = len(l[0])
        k = len(l)
        kI = np.identity(n, dtype=complex)
        for i in range(n):
            kI[i][i] = kI[i][i] * k
        
        sum = np.zeros((n,n), dtype=complex)
        for m in l:
            #print(str(m))
            sum = sum + m
        
        result = np.identity(n) - ComplexMatrix.supp(kI - sum, epsilonZ, epsilonG)
        return result


    @staticmethod
    def subsetCheckForProjections(p, q, whyWrong, epsilon):
        diff = (q @ p) - p
        n1 = ComplexMatrix.norm1(diff)
        if whyWrong and n1 > epsilon:
            print(str(p) + "is different from\n" + str(q))
        return n1 < epsilon