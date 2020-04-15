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
5: Vplc      12.0
6: Kip3      9.0
7: KoffIP3            5
8: a                  6
9: dinh      16.0
10: Ke                7
11: Be       70.0
12: d1                8
13: d5                9
14: epr               10
15: eta1              11
16: eta2              12
17: eta3              13
18: c0       39.0
19: k3                14
*/
functions {
    real[] sho(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
        real dydt[4];
        real beta;
        real m_inf;

        dydt[1] = theta[1] * theta[2] * exp(-theta[3] * t) - theta[4] * y[1];
        dydt[2] = (12.0 * y[1] * y[1]) / (81.0 + y[1] * y[1]) - theta[5] * y[2];
        dydt[3] = theta[6] * (y[4] + 16.0) * (16.0 / (y[4] * 16.0) - y[3]);
        beta = 1 + theta[7] * 70.0 / pow(theta[7] + y[4], 2);
        m_inf = y[2] * y[4] / ((theta[8] + y[2]) * (theta[9] + y[4]));
        dydt[4] = 1 / beta * (
            theta[10]
                * (theta[11] * pow(m_inf, 3) * pow(y[3], 3) + theta[12])
                * (39.0 - (1 + theta[10]) * y[4])
            - theta[13] * pow(y[4], 2) / (pow(theta[14], 2) + pow(y[4], 2))
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
    real mu_prior[14];     // mean of prior
    real sigma_prior[14];  // standard deviation of prior
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
    real<lower=0> sigma;
    real<lower=0> theta[14];
}
model {
    real y_hat[T, N];
    sigma ~ cauchy(0, 0.05);
    for (j in 1:14) {
        theta[j] ~ normal(mu_prior[j], sigma_prior[j]);
    }
    y_hat = integrate_ode_rk45(sho, y0, t0, ts, theta, x_r, x_i);
    for (t in 1:T) {
        y[t] ~ normal(y_hat[t, 4], sigma);
    }
}
