//old this can only handle 1 x observation
data {
    int<lower=0> N_obs;
    int<lower=0> N_cens;
    real x_obs[N_obs+N_cens];
    real y_obs[N_obs];
    real<lower=max(y_obs)> U;
    
}

transformed data {}
parameters {
    real<lower=U> y_cens[N_cens];
    real mu;
    real alpha;
    real<lower=0> sigma;
}

model {
    alpha ~ normal(0,10);
    mu ~ normal(0,10);
    sigma ~ normal(0,10);
    
    for (i in 1:N_obs){
        y_obs[i] ~ normal(mu+alpha * x_obs[i], sigma);
    }
    for (i in 1:N_cens){
        y_cens[i] ~ normal(mu+ alpha*x_obs[i+N_obs], sigma);
    }
}

generated quantities {
}