import math
import numpy as np
import galois 
import random


class FiniteField:
    def __init__(self, p, n):
        """
        Initialize the finite field GF(p^n).
        p: Prime number (characteristic of the field).
        n: Degree of the field extension (default is 1 for GF(p)).
        """
        self.p = p
        self.n = n
        self.q = p**n
        self.field = galois.GF(p**n)  # Create the finite field
        self.field.repr("poly")
        self.generator = self.field.primitive_element  # Find the primitive element
        self.polynomial_basis = self._generate_polynomial_basis()  # Generate polynomial basis
        self.dual_basis = self._generate_dual_basis()  # Generate dual basis
        # Instance-level cache for all SL(2, F) elements
        self._sl2_elements = None


    def _generate_polynomial_basis(self):
        """
        Generate the polynomial basis {1, a, a^2, ..., a^(n-1)} where a is the primitive element.
        """
        a = self.generator
        return [a**i for i in range(self.n)]

    def _generate_dual_basis(self):
        """
        Generate the dual basis of the polynomial basis.
        The dual basis {b_0, b_1, ..., b_(n-1)} satisfies Tr(b_i * a^j) = δ_ij (Kronecker delta).
        Uses galois.FieldArray for stable computation.
        """
        n = self.n
        F = self.field
        a = self.generator
        
        # Build Gram matrix G where G[i,j] = Tr(a^i * a^j)
        G = F(np.zeros((n, n), dtype=int))
        for i in range(n):
            for j in range(n):
                G[i, j] = self.trace(a**i * a**j)
        
        # Invert Gram matrix in the field
        G_inv = np.linalg.inv(G)
        
        # Dual basis: b_k has coefficients from k-th row of G_inv
        dual_basis = []
        for k in range(n):
            b_k = F(0)
            for j in range(n):
                b_k += G_inv[k, j] * a**j
            dual_basis.append(b_k)

        return dual_basis

    def trace(self, element):
        """
        Compute the trace of an element in the finite field.
        The trace is the sum of the element's conjugates: element + element^p + element^(p^2) + ... + element^(p^(n-1)).
        """
        trace = element
        current = element
        for _ in range(1 , self.n):
            current = current**self.p  # Raise to the power of p
            trace += current
        return trace

    def express_in_polynomial_basis(self, element):
        """
        Express an element in the polynomial basis {1, a, a^2, ..., a^(n-1)}.
        """
        coefficients = []
        for basis_element in self.dual_basis:
            coefficients.append(self.trace(element * basis_element))
        return coefficients

    def express_in_dual_basis(self, element):
        """
        Express an element in the dual basis.
        """
        coefficients = []
        for basis_element in self.polynomial_basis:
            coefficients.append(self.trace(element * basis_element))
        return coefficients

    def get_sl2_elements(self):
        """
        Instance-level cache: compute and return all elements of SL(2, F) once per instance.
        """
        if self._sl2_elements is not None:
            return self._sl2_elements

        F = self.field
        elements = []
        for a in F.elements:
            for b in F.elements:
                if a == 0 and b == 0:
                    continue
                if a != 0:
                    c0 = F(0)
                    d0 = a**-1
                else:
                    # a == 0, so b != 0
                    c0 = b**-1
                    d0 = F(0)
                for t in F.elements:
                    c = c0 + t * a
                    d = d0 + t * b
                    mat = F([[a, b], [c, d]])
                    elements.append(mat)

        self._sl2_elements = elements
        return self._sl2_elements

    def clear_sl2_elements(self):
        """Clear the cached SL(2, F) elements for this instance."""
        self._sl2_elements = None


def calculate_max_steps(q):
    """
    Calculate appropriate max_steps for SL(2,q) random walk to achieve near-uniform sampling.
    
    Args:
        q: Order of the finite field (prime power)
        
    Returns:
        Recommended number of steps for the random walk
    """
    # Base-2 logarithm of q
    log2_q = math.log2(q)
    
    # We use 3-5 steps per bit of q's representation as a conservative estimate
    return max(10, int(5 * log2_q))  # Ensure at least 10 steps for small q

def determinant(M):
    return M[0,0]*M[1,1]-M[0,1]*M[1,0]


# def sample_SL2(finite_field):
#     """
#     Sample a random element from SL(2,q) using a random walk approach.
    
#     Args:
#         q: The order of the finite field (must be a prime power)
#         max_steps: Maximum number of steps in the random walk
        
#     Returns:
#         A 2x2 matrix in SL(2,q)
#     """
#     # Create the finite field GF(q)
#     p = finite_field.p
#     n = finite_field.n
#     q = finite_field.q
#     F = finite_field.field
#     a = finite_field.generator
    
#     # Start with the identity matrix
#     current = F([[1, 0], [0, 1]])
#     max_steps = calculate_max_steps(q)
#     J = F([[0, 1], [p-1, 0]])
#     S = F([[1, a], [0, 1]])
#     G = F([[a**-1, 0], [0, a]])
#     # Perform random walk by multiplying by random generators
#     for _ in range(max_steps):
#         # Choose a random generator (could be any set of generators for SL(2,q))
#         generator = random.choice([J, S, G])
#         # Multiply current matrix by generator
#         current = current@generator

#     # Verify determinant is 1 (should always be true since we're multiplying elements of SL(2,q))
#     det = determinant(current)
#     assert det == F(1), "Generated matrix is not in SL(2,q)"
#     return current

def sample_SL2(finite_field):
    """
    Uniformly sample from SL(2, F), where F = GF(2^n) (or any GF(q)).
    Returns a 2x2 matrix over F with det = 1.
    """
    F = finite_field.field
    # 1) uniform nonzero (a,b)
    while True:
        a = F.Random()
        b = F.Random()
        if a != 0 or b != 0:
            break

    # 2) pick a particular solution (c0,d0)
    if a != 0:
        c0 = F(0)
        d0 = a**-1
    else:
        # a == 0, so b != 0
        c0 = b**-1
        d0 = F(0)

    # 3) uniform t, generate (c,d)
    t = F.Random()
    c = c0 + t*a
    d = d0 + t*b
    current = F([[a, b], [c, d]])
    det = determinant(current)
    assert det == F(1), "Generated matrix is not in SL(2,q)"
    return current

def sample_SL2_noidentity(finite_field):
    """
    Uniformly sample from SL(2, F), where F = GF(2^n) (or any GF(q)).
    Returns a 2x2 matrix over F with det = 1.
    """
    F = finite_field.field
    # 1) uniform nonzero (a,b)
    while True:
        a = F.Random()
        b = F.Random()
        if a != 0 or b != 0:
            break

    # 2) pick a particular solution (c0,d0)
    if a != 0:
        c0 = F(0)
        d0 = a**-1
    else:
        # a == 0, so b != 0
        c0 = b**-1
        d0 = F(0)

 
    # 3) uniform t, generate (c,d)
    t = F.Random()
    if a == F(1) and b == F(0):
        while t == F(0):
            t = F.Random()
    c = c0 + t*a
    d = d0 + t*b


    current = F([[a, b], [c, d]])
    det = determinant(current)
    assert det == F(1), "Generated matrix is not in SL(2,q)"
    assert not (a == F(1) and b == F(0) and c == F(0) and d == F(1)), "Generated identity matrix!"
    return current

def random_SL_transformation(finite_field,XvList,ZvList):
    F = finite_field.field
    n = finite_field.n

    random_element = sample_SL2(finite_field)
    new_XvList, new_ZvList = [], []
    for i in range(len(XvList)): 
        Xv,Zv = XvList[i], ZvList[i]
        x, z = F(0), F(0)
        for j in range(n):
            x += finite_field.polynomial_basis[j]*Xv[j]
            z += finite_field.dual_basis[j]*Zv[j]
        v = F([x,z])
        new_v = random_element@v
        new_Xv = list(finite_field.express_in_polynomial_basis(new_v[0]))
        new_Zv = list(finite_field.express_in_dual_basis(new_v[1]))
        new_XvList.append(new_Xv)
        new_ZvList.append(new_Zv)
    return new_XvList, new_ZvList

def random_SL_transformation_noidentity(finite_field,XvList,ZvList):
    F = finite_field.field
    n = finite_field.n

    random_element = sample_SL2_noidentity(finite_field)
    new_XvList, new_ZvList = [], []
    for i in range(len(XvList)): 
        Xv,Zv = XvList[i], ZvList[i]
        x, z = F(0), F(0)
        for j in range(n):
            x += finite_field.polynomial_basis[j]*Xv[j]
            z += finite_field.dual_basis[j]*Zv[j]
        v = F([x,z])
        new_v = random_element@v
        new_Xv = list(finite_field.express_in_polynomial_basis(new_v[0]))
        new_Zv = list(finite_field.express_in_dual_basis(new_v[1]))
        new_XvList.append(new_Xv)
        new_ZvList.append(new_Zv)
    return new_XvList, new_ZvList

def all_SL2_elements(finite_field):
    """Backward-compatible wrapper to use instance-level cache."""
    return finite_field.get_sl2_elements()

def SL_transformation_all(finite_field,XvList,ZvList):
    F = finite_field.field
    n = finite_field.n

    SL_elements = all_SL2_elements(finite_field)
    new_XvLists, new_ZvLists = [], []
    for random_element in SL_elements:
        new_XvList, new_ZvList = [], []
        for i in range(len(XvList)): 
            Xv,Zv = XvList[i], ZvList[i]
            x, z = F(0), F(0)
            for j in range(n):
                x += finite_field.polynomial_basis[j]*Xv[j]
                z += finite_field.dual_basis[j]*Zv[j]
            v = F([x,z])
            new_v = random_element@v
            new_Xv = list(finite_field.express_in_polynomial_basis(new_v[0]))
            new_Zv = list(finite_field.express_in_dual_basis(new_v[1]))
            new_XvList.append(new_Xv)
            new_ZvList.append(new_Zv)
        new_XvLists.append(new_XvList)
        new_ZvLists.append(new_ZvList)
    return new_XvLists, new_ZvLists