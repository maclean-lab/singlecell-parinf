/*
variables by index
1: PLC
2: IP3
3: h
4: ca
parameter list
   name      value    index
1: KonATP             1
2: L                  2
3: Katp               3
4: KoffPLC            4
5: Vplc               5
6: Kip3_sq            6
7: KoffIP3            7
8: a                  8
9: dinh               9
10: Ke                10
11: Be                11
12: d1       0.0
13: d5                12
14: epr               13
15: eta1              14
16: eta2              15
17: eta3              16
18: c0                17
19: k3_sq             18
*/
functions {
    real[] sho(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
        real dydt[4];
        real beta;
        real m_inf;

        dydt[1] = theta[1] * theta[2] * exp(-theta[3] * t) - theta[4] * y[1];
        dydt[2] = (theta[5] * y[1] * y[1])
            / (theta[6] + y[1] * y[1]) - theta[7] * y[2];
        dydt[3] = theta[8] * (theta[9] - (y[4] + theta[9]) * y[3]);
        beta = pow(theta[10] + y[4], 2)
            / (pow(theta[10] + y[4], 2) + theta[10] * theta[11]);
        m_inf = y[4] / (theta[12] + y[4]);
        dydt[4] = beta * (
            theta[13]
                * (theta[14] * pow(m_inf, 3) * pow(y[3], 2) + theta[15])
                * (theta[17] - (1 + theta[13]) * y[4])
            - theta[16] * y[4] * y[4] / (theta[18] + y[4] * y[4])
        );

        return dydt;
    }
}
data {
    int<lower=1> N;        // number of variables
    int<lower=1> T;        // number of time steps
    real y0[N];            // initial values
    real y[T];             // values at all time points
    real t0;               // initial time point
    real ts[T];            // all time points
    real mu_prior[18];     // mean of prior
    real sigma_prior[18];  // standard deviation of prior
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
    real<lower=0> sigma;
    real<lower=0> theta[18];
}
model {
    real y_hat[T, N];
    sigma ~ cauchy(0, 0.05);
    for (j in 1:18) {
        theta[j] ~ normal(mu_prior[j], sigma_prior[j]);
    }
    y_hat = integrate_ode_rk45(sho, y0, t0, ts, theta, x_r, x_i);
    for (t in 1:T) {
        y[t] ~ normal(y_hat[t, 4], sigma);
    }
}
