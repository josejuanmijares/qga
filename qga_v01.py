import matplotlib.pyplot as plt
import numpy as np
import qutip as qc
import timeit
import pickle


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


# single gene
def CO_rx(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.rx(phase).full()))


def CO_ry(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.ry(phase).full()))


def CO_rz(Chromosome, phase=np.pi / 8):
    return np.dot(Chromosome, np.array(qc.rz(phase).full()))


# Double genes
def CO_swap_alpha(Chromosome, phase=np.pi / 8):
    l = Chromosome.shape[0] // 2
    for k0 in range(l):
        C0 = Chromosome[2 * k0, :]
        C1 = Chromosome[2 * k0 + 1, :]
        C = np.hstack((C0, C1))
        C = np.dot(np.array(qc.swapalpha(phase).full()), C)
        C = C.reshape((-1, 2))
        Chromosome[2 * k0, :] = C[:, 0]
        Chromosome[2 * k0 + 1, :] = C[:, 1]
    return Chromosome

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


# Single gene
def MU_Pauli_z(Chromosome):
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.dot(Chromosome, pauli_z)


def MU_Pauli_y(Chromosome):
    pauli_y = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
    return np.dot(Chromosome, pauli_y)


def MU_Pauli_x(Chromosome):
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    return np.dot(Chromosome, pauli_x)


def MU_Phase_shift(Chromosome, phase=np.pi / 8.0):
    r = np.array(qc.phasegate(phase).full())
    return np.dot(Chromosome, r)


def MU_rot(Chromosome, phi):
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    l = Chromosome.shape[0]
    for k0 in range(l):
        C = Chromosome[k0,:]
        C = np.dot(C,np.array(qc.phasegate(phi).full()))
        C = np.array(np.dot(C, pauli_x))

        Chromosome[k0,:] = C
    return Chromosome


# Double genes
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
        C = np.vstack((C0, C1))
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


def test(f0, f0arg, f1, f1arg, Niter, NGenes, Ch0):
    Ch = Ch0.copy()
    l = 0
    for k in range(Niter):
        # measurement
        mask0, mask1, val = measure_chromosome(Ch, NGenes)

        # Cross-Over
        if np.sum(mask0) > 1:
            Ch[mask0, :] = eval(f0 + '(' + f0arg + ')')

        # Mutation
        if np.sum(mask1) > 1:
            if l == 0:
                l = 1
            Ch[mask1, :] = eval(f1 + '(' + f1arg + ')')

    return calc_PSDsigma_mu(val)


if __name__ == '__main__':

    # initialization
    NGenes = 128
    NIterations = 1000
    Chromosome = chromosome_initialization(NGenes)
    # print("####################### initialization")
    # print(Chromosome)
    phase = np.array([0.125, 0.25, 0.5, 0.75, 0.875]) * np.pi
    CO_p = [None, None, *phase, *phase, *phase, *phase]
    MU_p = [None, None, None, None, None, None, *phase, *phase]

    CO_function = ['CO_swap', 'CO_iswap', 'CO_swap_alpha', 'CO_swap_alpha', 'CO_swap_alpha', 'CO_swap_alpha',
                   'CO_swap_alpha',
                   'CO_rx', 'CO_rx', 'CO_rx', 'CO_rx', 'CO_rx',
                   'CO_ry', 'CO_ry', 'CO_ry', 'CO_ry', 'CO_ry',
                   'CO_rz', 'CO_rz', 'CO_rz', 'CO_rz', 'CO_rz']

    MU_function = ['MU_Pauli_x', 'MU_Pauli_y', 'MU_Pauli_z', 'MU_sqrtswap', 'MU_sqrtiswap', 'MU_sqrtnot',
                   'MU_Phase_shift', 'MU_Phase_shift', 'MU_Phase_shift', 'MU_Phase_shift', 'MU_Phase_shift', 'MU_rot',
                   'MU_rot', 'MU_rot', 'MU_rot', 'MU_rot']

    sigma_array = np.zeros((22, 16))
    mu_array = np.zeros((22, 16))
    for i, (f0, Ang0) in enumerate(zip(CO_function, CO_p)):
        for j, (f1, Ang1) in enumerate(zip(MU_function, MU_p)):
            if Ang0:
                f0arg = 'Ch[mask0,:], ' + str(Ang0)
            else:
                f0arg = 'Ch[mask0,:]'

            if Ang1:
                f1arg = 'Ch[mask1,:], ' + str(Ang1)
            else:
                f1arg = 'Ch[mask1,:]'
            start = timeit.timeit()
            sigma_array[i, j], mu_array[i, j] = test(f0, f0arg, f1, f1arg,  NIterations, NGenes, Chromosome)
            print([i, j, f0, f1, np.abs(timeit.timeit() - start)])
    print(sigma_array)

    file = open('data_1000.pkl','wb')
    pickle.dump((sigma_array,mu_array),file)
    file.close()


    #
    # track_mean = np.zeros(NIterations)
    # track_std = np.zeros(NIterations)
    # track_PSDstd = np.zeros(NIterations)
    # track_PSDmu = np.zeros(NIterations)
    # for k in range(NIterations):
    #     print("####################################### k=%d"%k)
    #     # measurement
    #     mask1, mask2, val = measure_chromosome(Chromosome, NGenes)
    #
    #     # rotation
    #     if np.any(mask1):
    #
    #
    #
    #
    #
    #
    #
    #         Chromosome[mask1, :] = CO_iswap(Chromosome[mask1, :])
    #
    #
    #
    #     # Chromosome[mask1,:] = CO_qubit_rotation(Chromosome[mask1,:])
    #     # print("### rotation / cross over ###")
    #     #print(Chromosome)
    #
    #     # mutation
    #     if np.any(mask2):
    #         Chromosome[mask2, :] = MU_qubit_rotation(Chromosome[mask2, :])  ## affects Beta principally
    #     # print("### mutation ###")
    #     #print(Chromosome)
    #
    #     # fitness
    #     # print("### fitness ###")
    #     track_mean[k] = np.mean(val)
    #     track_std[k] = np.std(val)
    #     track_PSDstd[k], track_PSDmu[k] = calc_PSDsigma_mu(val)
    #
    #     print("mean = ", str(track_mean[k]))
    #     print("std = ", str(track_std[k]))
    #     print("PSDstd = ", str(track_PSDstd[k]))
    #     print("PSDmu = ", str(track_PSDmu[k]))
    #
    # plt.plot(np.arange(0, len(track_mean), 1), track_mean, 'b.-')
    # plt.plot(np.arange(0, len(track_std), 1), track_std, 'r.-')
    # plt.plot(np.arange(0, len(track_PSDstd), 1), track_PSDstd, 'g.-')
    # plt.plot(np.arange(0, len(track_PSDmu), 1), track_PSDmu, 'm.-')
    # plt.show()
    #
    #     # print(Chromosome)
    #
    #     # ket = (qc.basis(2,0) + qc.basis(2,1)).unit()   # the superposition of 2 states
    #     # print(ket)
