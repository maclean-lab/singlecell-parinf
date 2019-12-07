/*
variables by index
1: RNA
2: p
paramters by index
1: betaM
2: alphaM
3: beta
4: alpha
*/
functions {
    real[] sho(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
        real dydt[2];

        dydt[1] = theta[1] - theta[2] * y[1];
        dydt[2] = theta[3] * y[1] - theta[4] * y[2];

        return dydt;
    }
}
data {
    int<lower=1> N;  // number of variables
    int<lower=1> T;  // number of time steps
    real y0[N];      // initial values
    real y[T];       // values at all time points
    real t0;         // initial time point
    real ts[T];      // all time points
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
    real<lower=0> sigma;
    real<lower=0> theta[4];
}
model {
    real y_hat[T, N];
    sigma ~ cauchy(0, 0.05);
    for (j in 1:4) {
        theta[j] ~ uniform(0.0, 10.0);
    }
    y_hat = integrate_ode_rk45(sho, y0, t0, ts, theta, x_r, x_i);
    for (t in 1:T) {
        y[t] ~ normal(y_hat[t, 2], sigma);
    }
}
