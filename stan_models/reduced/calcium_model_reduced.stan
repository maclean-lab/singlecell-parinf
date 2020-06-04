/*
variables by index
1: h
2: ca
parameters by index
1: a
2: dinh
3: Ke
4: Be
5: d1
6: d5
7: epr
8: eta1
9: eta2
10: eta3
11: c0
12: k3_sq
*/
functions {
    real[] sho(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
        real dydt[2];
        real beta;
        real m_inf;

        dydt[1] = theta[1] * (theta[2] - (y[2] + theta[2]) * y[1]);
        beta = pow(theta[3] + y[2], 2)
            / (pow(theta[3] + y[2], 2) + theta[3] * theta[4]);
        m_inf = theta[5] * y[2] / (theta[6] + y[2]);
        dydt[2] = beta * (
            theta[7]
                * (theta[8] * pow(m_inf, 3) * pow(y[1], 3) + theta[9])
                * (theta[11] - (1 + theta[7]) * y[2])
            - theta[10] * y[2] * y[2] / (theta[12] + y[2] * y[2])
        );

        return dydt;
    }
}
data {
    int<lower=1> N;           // number of variables
    int<lower=1> T;           // number of time steps
    int<lower=1> num_params;  // number of parameters
    real y0[N];               // initial values
    real y[T];                // values at all time points
    real t0;                  // initial time point
    real ts[T];               // all time points
    real mu_prior[12];        // mean of prior
    real sigma_prior[12];     // standard deviation of prior
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
    real<lower=0> sigma;
    real<lower=0> theta[num_params];
}
model {
    real y_hat[T, N];
    sigma ~ cauchy(0, 0.05);
    for (j in 1:num_params) {
        theta[j] ~ normal(mu_prior[j], sigma_prior[j]);
    }
    y_hat = integrate_ode_rk45(sho, y0, t0, ts, theta, x_r, x_i);
    for (t in 1:T) {
        y[t] ~ normal(y_hat[t, 2], sigma);
    }
}
