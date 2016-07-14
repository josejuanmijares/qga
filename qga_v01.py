import matplotlib.pyplot as plt
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


def CO_swap(Chromosome):
    l = Chromosome.shape[0] // 2
    for k0 in range(l):
        C0 = Chromosome[2 * k0, :]
        C1 = Chromosome[2 * k0 + 1, :]
        C = np.hstack((C0, C1))
        C = np.dot(np.array(qc.swap().full()), C)
        C = C.reshape((-1, 2))
        Chromosome[2 * k0, :] = C[:, 0]
        Chromosome[2 * k0 + 1, :] = C[:, 1]
    return Chromosome


def CO_iswap(Chromosome):
    l = Chromosome.shape[0] // 2
    for k0 in range(l):
        C0 = Chromosome[2 * k0, :]
        C1 = Chromosome[2 * k0 + 1, :]
        C = np.hstack((C0, C1))
        C = np.dot(np.array(qc.iswap().full()), C)
        C = C.reshape((-1, 2))
        Chromosome[2 * k0, :] = C[:, 0]
        Chromosome[2 * k0 + 1, :] = C[:, 1]
    return Chromosome

def MU_sqrtswap(Chromosome):
    l = Chromosome.shape[0] // 2
    for k0 in range(l):
        C0 = Chromosome[2 * k0, :]
        C1 = Chromosome[2 * k0 + 1, :]
        C = np.hstack((C0, C1))
        C = np.dot(np.array(qc.sqrtswap().full()), C)
        C = C.reshape((-1, 2))
        Chromosome[2 * k0, :] = C[:, 0]
        Chromosome[2 * k0 + 1, :] = C[:, 1]
    return Chromosome


def MU_sqrtiswap(Chromosome):
    l = Chromosome.shape[0] // 2
    for k0 in range(l):
        C0 = Chromosome[2 * k0, :]
        C1 = Chromosome[2 * k0 + 1, :]
        C = np.hstack((C0, C1))
        C = np.dot(np.array(qc.sqrtiswap().full()), C)
        C = C.reshape((-1, 2))
        Chromosome[2 * k0, :] = C[:, 0]
        Chromosome[2 * k0 + 1, :] = C[:, 1]
    return Chromosome


def MU_sqrtnot(Chromosome):
    l = Chromosome.shape[0] // 2
    for k0 in range(l):
        C0 = Chromosome[2 * k0, :]
        C1 = Chromosome[2 * k0 + 1, :]
        C = np.hstack((C0, C1))
        C = np.dot(np.array(qc.sqrtnot().full()), C)
        C = C.reshape((-1, 2))
        Chromosome[2 * k0, :] = C[:, 0]
        Chromosome[2 * k0 + 1, :] = C[:, 1]
    return Chromosome



def calc_PSDsigma_mu(X):
    X = X - np.mean(X)
    N = len(X)
    sigma = np.std(X)
    Fr = np.fft.fft(X)
    Frconj = np.conj(Fr)
    Sxx = Fr * Frconj / ((2 * N - 1) * (sigma ** 2.0))
    return np.std(np.abs(Sxx)), np.mean(np.abs(Sxx))

def MU_rot(Chromosomes,phi):
    pauli_x = [[0,1],[1,0]]
    return np.dot(np.dot(Chromosomes, qc.phasegate(phi)),pauli_x)

if __name__ == '__main__':

    # initialization
    NGenes = 128
    NIterations = 1000
    Chromosome = chromosome_initialization(NGenes)
    # print("####################### initialization")
    # print(Chromosome)

    track_mean = np.zeros(NIterations)
    track_std = np.zeros(NIterations)
    track_PSDstd = np.zeros(NIterations)
    track_PSDmu = np.zeros(NIterations)
    for k in range(NIterations):
        print("####################################### k=%d"%k)
        # measurement
        mask1, mask2, val = measure_chromosome(Chromosome, NGenes)
        # print("### mask ###")
        #print(mask1, "____", mask2)

        # rotation
        if np.any(mask1):
            Chromosome[mask1, :], val = CO_iswap(Chromosome[mask1, :], val)
            # Chromosome[mask1, :], val = CO_swap(Chromosome[mask1, :], val)      ## only affects Beta
            ## PSD goes from sigma 0.75 to 2.5
            ## mean 0.5



        # Chromosome[mask1,:] = CO_qubit_rotation(Chromosome[mask1,:])
        # print("### rotation / cross over ###")
        #print(Chromosome)

        # mutation
        if np.any(mask2):
            Chromosome[mask2, :] = MU_qubit_rotation(Chromosome[mask2, :])  ## affects Beta principally
        # print("### mutation ###")
        #print(Chromosome)

        # fitness
        # print("### fitness ###")
        track_mean[k] = np.mean(val)
        track_std[k] = np.std(val)
        track_PSDstd[k], track_PSDmu[k] = calc_PSDsigma_mu(val)

        print("mean = ", str(track_mean[k]))
        print("std = ", str(track_std[k]))
        print("PSDstd = ", str(track_PSDstd[k]))
        print("PSDmu = ", str(track_PSDmu[k]))

    plt.plot(np.arange(0, len(track_mean), 1), track_mean, 'b.-')
    plt.plot(np.arange(0, len(track_std), 1), track_std, 'r.-')
    plt.plot(np.arange(0, len(track_PSDstd), 1), track_PSDstd, 'g.-')
    plt.plot(np.arange(0, len(track_PSDmu), 1), track_PSDmu, 'm.-')
    plt.show()

        # print(Chromosome)

        # ket = (qc.basis(2,0) + qc.basis(2,1)).unit()   # the superposition of 2 states
        # print(ket)
