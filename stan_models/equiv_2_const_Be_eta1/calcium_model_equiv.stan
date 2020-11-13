/*
variables by index
1: PLC
2: IP3
3: h
4: ca
parameters by index
1: L
2: Katp
3: KoffPLC
4: Vplc
5: Kip3_sq
6: KoffIP3
7: a
8: dinh
9: Ke
10: d1
11: d5
12: epr
13: eta2
14: eta3
15: c0
16: k3_sq
*/
functions {
    real[] sho(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
        real dydt[4];
        real beta;
        real m_inf;

        dydt[1] = theta[1] * exp(-theta[2] * t) - theta[3] * y[1];
        dydt[2] = (theta[4] * y[1] * y[1])
            / (theta[5] + y[1] * y[1]) - theta[6] * y[2];
        dydt[3] = theta[7] * (theta[8] - (y[4] + theta[8]) * y[3]);
        beta = pow(theta[9] + y[4], 2)
            / (pow(theta[9] + y[4], 2) + theta[9] * 150);
        m_inf = y[2] * y[4] / ((theta[10] + y[2]) * (theta[11] + y[4]));
        dydt[4] = beta * (
            theta[12]
                * (575 * pow(m_inf, 3) * pow(y[3], 3) + theta[13])
                * (theta[15] - (1 + theta[12]) * y[4])
            - theta[14] * y[4] * y[4] / (theta[16] + y[4] * y[4])
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
    real mu_prior[16];     // mean of prior
    real sigma_prior[16];  // standard deviation of prior
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
    real<lower=0> sigma;
    real<lower=0> theta[16];
}
model {
    real y_hat[T, N];
    sigma ~ cauchy(0, 0.05);
    for (j in 1:16) {
        theta[j] ~ normal(mu_prior[j], sigma_prior[j]);
    }
    y_hat = integrate_ode_rk45(sho, y0, t0, ts, theta, x_r, x_i);
    for (t in 1:T) {
        y[t] ~ normal(y_hat[t, 4], sigma);
    }
}
