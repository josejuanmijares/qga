import numpy as np
import qutip as qc


def chromosome_initialization(NGenes: int = 1, nbits=1):
    temp = qc.basis(nbits + 1, 0)
    for k in range(1, 2 ** nbits):
        temp += qc.basis(nbits + 1, k)
    temp = np.array((temp.unit()).full()).flatten()
    return np.array([temp for k in range(NGenes)]).reshape((-1, 2))


def measure_chromosome(Chromosome, NGenes: int = 1):
    mask1 = []
    mask2 = []
    val = []
    for c in Chromosome[:, 0]:
        r = np.random.random()
        if r > (c.real ** 2.0):
            mask1.append(True)
        else:
            mask1.append(False)
        if r < 0.02:
            mask2.append(True)
        else:
            mask2.append(False)
        val.append(r)
    mask1 = np.array(mask1, dtype=bool)
    mask2 = np.array(mask2, dtype=bool)
    val = np.array(val)
    return mask1, mask2, val


def MU_qubit_rotation(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.phasegate(phase).full()))


def CO_swap_alpha(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.swapalpha(phase).full()))


def CO_rx(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.rx(phase).full()))


def CO_ry(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.ry(phase).full()))


def CO_rz(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.rz(phase).full()))


def CO_swap(Chromosome, val):
    l = Chromosome.shape[0] // 2
    for k0 in range(l):
        C0 = Chromosome[2 * k0, :]
        C1 = Chromosome[2 * k0 + 1, :]
        C = np.hstack((C0, C1))
        v = np.hstack(([val[2 * k0], 0], [val[2 * k0 + 1], 0]))
        C = np.dot(np.array(qc.swap().full()), C)
        v = np.dot(np.array(qc.swap().full()), v)

        C = C.reshape((-1, 2))
        Chromosome[2 * k0, :] = C[:, 0]
        Chromosome[2 * k0 + 1, :] = C[:, 1]
        val[2 * k0] = v[0]
        val[2 * k0 + 1] = v[1]

    return Chromosome, val


def MU_iswap(Chromosome):
    return np.dot(Chromosome, np.array(qc.iswap().full()))


def MU_sqrtswap(Chromosome):
    return np.dot(Chromosome, np.array(qc.sqrtswap().full()))


def MU_sqrtiswap(Chromosome):
    return np.dot(Chromosome, np.array(qc.sqrtiswap().full()))


def MU_sqrtnot(Chromosome):
    return np.dot(Chromosome, np.array(qc.sqrtnot().full()))

if __name__ == '__main__':

    # initialization
    NGenes = 4
    Chromosome = chromosome_initialization(NGenes)
    print("####################### initialization")
    print(Chromosome)

    for k in range(100):
        print("####################################### k=", str(k))
        # measurement
        mask1, mask2, val = measure_chromosome(Chromosome, NGenes)
        print("### mask ###")
        print(mask1, "____", mask2)

        # rotation
        Chromosome[mask1, :], val = CO_swap(Chromosome[mask1, :], val)
        # Chromosome[mask1,:] = CO_qubit_rotation(Chromosome[mask1,:])
        print("### rotation / cross over ###")
        print(Chromosome)

        # mutation
        Chromosome[mask2, :] = MU_qubit_rotation(Chromosome[mask2, :])
        print("### mutation ###")
        print(Chromosome)

        # fitness
        print("### fitness ###")
        print("mean = ", str(np.mean(val)))
        print("std = ", str(np.std(val)))




        # print(Chromosome)

        # ket = (qc.basis(2,0) + qc.basis(2,1)).unit()   # the superposition of 2 states
        # print(ket)
