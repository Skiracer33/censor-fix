//this can only handler right censoring
data {
    int<lower=0> N_obs; //number of observations
    int<lower=0> N_cens; //number of censored values
    int<lower=0> K; //number of predictors
    matrix[N_obs+N_cens, K] x_obs;
    real y_obs[N_obs];
    real<lower=max(y_obs)> U;
    
}

transformed data {}
parameters {
    real<lower=U> y_cens[N_cens];
    real mu;
    vector[K] alpha;
    real<lower=0> sigma;
}

model {
    alpha ~ normal(0,10);
    mu ~ normal(0,10);
    sigma ~ normal(0,10);
    
    for (i in 1:N_obs){
        y_obs[i] ~ normal(mu + x_obs[i] * alpha , sigma);
    }
    for (i in 1:N_cens){
        y_cens[i] ~ normal(mu + x_obs[i+N_obs] * alpha, sigma);
    }
}

generated quantities {
}