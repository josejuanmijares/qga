import numpy as np
import qutip as qc


def chromosome_initialization(NGenes: np.int = 1):
    return [(qc.basis(2, 0) + qc.basis(2, 1)).unit() for k in range(NGenes)]
    # qc.hadamard_transform(NGenes)


def measure_chromosome(Chromosome, NGenes: np.int = 1):
    return np.array([np.random.random() > (np.real(c[:])[0] ** 2.) for c in Chromosome], dtype=np.int)


def qubit_rotation(Chromosome, phase=np.pi / 8):
    return qc.phasegate(phase)*Chromosome


if __name__ == '__main__':

    # initialization
    NGenes = np.int(20.)
    Chromosome = chromosome_initialization(NGenes)
    print("####################### initialization")
    print(Chromosome)

    for k in range(100):
        print("####################################### k=", str(k))
        # measurement
        mask = np.array(measure_chromosome(Chromosome, NGenes).flatten(), dtype=np.bool)
        print("### mask ###")
        print(mask)

        # rotation
        Chromosome = np.array(Chromosome)
        Chromosome[mask] = qubit_rotation(Chromosome[mask])
        print("### rotation / cross over ###")
        print(Chromosome)



        # print(Chromosome)

        # ket = (qc.basis(2,0) + qc.basis(2,1)).unit()   # the superposition of 2 states
        # print(ket)
