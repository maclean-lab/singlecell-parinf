import numpy as np

def calcium_ode_vanilla(t, y, theta):
    """Original calcium model from Yao 2016"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] * theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (y[3] + theta[8]) \
        * (theta[8] / (y[3] * theta[8]) - y[2])
    beta_inv = 1 + theta[9] * theta[10] / np.power(theta[9] + y[3], 2)
    m_inf = y[1] * y[3] / ((theta[11] + y[1]) * (theta[12] + y[3]))
    dydt[3] = 1 / beta_inv * (
        theta[13]
            * (theta[14] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[15])
            * (theta[17] - (1 + theta[13]) * y[3])
        - theta[16] * y[3] * y[3] / (np.power(theta[18], 2) + y[3] * y[3])
    )

    return dydt

def calcium_ode_equiv_1(t, y, theta):
    """Calcium model with equivalent ODEs"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (theta[8] - (y[3] + theta[8]) * y[2])
    beta = np.power(theta[9] + y[3], 2) \
        / (np.power(theta[9] + y[3], 2) + theta[9] * theta[10])
    m_inf = y[1] * y[3] / ((theta[11] + y[1]) * (theta[12] + y[3]))
    dydt[3] = beta * (
        theta[13]
            * (theta[14] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[15])
            * (theta[17] - (1 + theta[13]) * y[3])
        - theta[16] * y[3] * y[3] / (theta[18] + y[3] * y[3])
    )

    return dydt

def calcium_ode_equiv_2(t, y, theta):
    """Calcium model with equivalent ODEs"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * np.exp(-theta[1] * t) - theta[2] * y[0]
    dydt[1] = (theta[3] * y[0] * y[0]) \
        / (theta[4] + y[0] * y[0]) - theta[5] * y[1]
    dydt[2] = theta[6] * (theta[7] - (y[3] + theta[7]) * y[2])
    beta = np.power(theta[8] + y[3], 2) \
        / (np.power(theta[8] + y[3], 2) + theta[8] * theta[9])
    m_inf = y[1] * y[3] / ((theta[10] + y[1]) * (theta[11] + y[3]))
    dydt[3] = beta * (
        theta[12]
            * (theta[13] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[14])
            * (theta[16] - (1 + theta[12]) * y[3])
        - theta[15] * y[3] * y[3] / (theta[17] + y[3] * y[3])
    )

    return dydt

def calcium_ode_equiv_2_const_eta1(t, y, theta):
    """Calcium model with equivalent ODEs"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * np.exp(-theta[1] * t) - theta[2] * y[0]
    dydt[1] = (theta[3] * y[0] * y[0]) \
        / (theta[4] + y[0] * y[0]) - theta[5] * y[1]
    dydt[2] = theta[6] * (theta[7] - (y[3] + theta[7]) * y[2])
    beta = np.power(theta[8] + y[3], 2) \
        / (np.power(theta[8] + y[3], 2) + theta[8] * theta[9])
    m_inf = y[1] * y[3] / ((theta[10] + y[1]) * (theta[11] + y[3]))
    dydt[3] = beta * (
        theta[12]
            * (575 * np.power(m_inf, 3) * np.power(y[2], 3) + theta[13])
            * (theta[15] - (1 + theta[12]) * y[3])
        - theta[14] * y[3] * y[3] / (theta[16] + y[3] * y[3])
    )

    return dydt

def calcium_ode_const_1(t, y, theta):
    """Calcium model with d_1 set to constants"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (theta[8] - (y[3] + theta[8]) * y[2])
    beta = np.power(theta[9] + y[3], 2) \
        / (np.power(theta[9] + y[3], 2) + theta[9] * theta[10])
    m_inf = y[3] / (theta[11] + y[3])
    dydt[3] = beta * (
        theta[12]
            * (theta[13] * np.power(m_inf, 3) * np.power(y[2], 2) + theta[14])
            * (theta[16] - (1 + theta[12]) * y[3])
        - theta[15] * y[3] * y[3] / (theta[17] + y[3] * y[3])
    )

    return dydt

def calcium_ode_const_2(t, y, theta):
    """Calcium model with d_1 and d_5 set to constants"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (theta[8] - (y[3] + theta[8]) * y[2])
    beta = np.power(theta[9] + y[3], 2) \
        / (np.power(theta[9] + y[3], 2) + theta[9] * theta[10])
    m_inf = y[1] * y[3] / ((0.13 + y[1]) * (0.0823 + y[3]))
    dydt[3] = beta * (
        theta[11]
            * (theta[12] * np.power(m_inf, 3) * np.power(y[2], 2) + theta[13])
            * (theta[15] - (1 + theta[11]) * y[3])
        - theta[14] * y[3] * y[3] / (theta[16] + y[3] * y[3])
    )

    return dydt

def calcium_ode_const_3(t, y, theta):
    """Calcium model with equivalent ODEs and d_1 set to constant"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * np.exp(-theta[1] * t) - theta[2] * y[0]
    dydt[1] = (theta[3] * y[0] * y[0]) \
        / (theta[4] + y[0] * y[0]) - theta[5] * y[1]
    dydt[2] = theta[6] * (theta[7] - (y[3] + theta[7]) * y[2])
    beta = np.power(theta[8] + y[3], 2) \
        / (np.power(theta[8] + y[3], 2) + theta[8] * theta[9])
    m_inf = y[1] * y[3] / ((0.13 + y[1]) * (theta[11] + y[3]))
    dydt[3] = beta * (
        theta[11]
            * (theta[12] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[13])
            * (theta[15] - (1 + theta[11]) * y[3])
        - theta[14] * y[3] * y[3] / (theta[16] + y[3] * y[3])
    )

    return dydt

def calcium_ode_reduced(t, y, theta):
    """Calcium model with h and Ca2+ only"""
    dydt = np.zeros(2)

    dydt[0] = theta[0] * (theta[1] - (y[1] + theta[1]) * y[0])
    beta = np.power(theta[2] + y[1], 2) \
        / (np.power(theta[2] + y[1], 2) + theta[2] * theta[3])
    m_inf = theta[4] * y[1] / (theta[5] + y[1])
    dydt[1] = beta * (
        theta[6]
            * (theta[7] * np.power(m_inf, 3) * np.power(y[0], 3) + theta[8])
            * (theta[10] - (1 + theta[6]) * y[1])
        - theta[9] * y[1] * y[1] / (theta[11] + y[1] * y[1])
    )

    return dydt

