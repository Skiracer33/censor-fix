<<<<<<< HEAD
data {
    int<lower=0> N_obs; //number of observations
    int<lower=0> N_cens_right; //number of censored right values
    int<lower=0> N_cens_left; //number of censored left values 
    int<lower=0> K; //number of predictors
    matrix[N_obs+N_cens_left+N_cens_right, K] x_obs;
    real y_obs[N_obs];
    real<lower=max(y_obs)> U;
    real<upper=min(y_obs)> L;  
}

transformed data {}
parameters {
    real<lower=U> y_cens_right[N_cens_right];
    real<upper=L> y_cens_left[N_cens_left];
    real mu;
    real xi;
    vector[K] alpha;
    real<lower=0> sigma;
}

model {
    alpha ~ normal(0,10);
    mu ~ normal(0,10);
    sigma ~ normal(0,10);
    xi ~ normal(0,10);
    
    for (i in 1:N_cens_left){
        y_cens_left[i] ~ skew_normal(xi, mu + x_obs[i] * alpha, sigma);
    }

    for (i in 1:N_obs){
        y_obs[i] ~ normal(xi, mu + x_obs[i+N_cens_left] * alpha , sigma);
    }

    for (i in 1:N_cens_right){
        y_cens_right[i] ~ normal(xi, mu + x_obs[i+N_obs+N_cens_left] * alpha, sigma);
    }
    
}

generated quantities {
}
=======
//this works with just one column
data {
  int<lower=0> N_obs;
  int<lower=0> N_cens;
  real y_obs[N_obs];
  real<lower=max(y_obs)> U;
}
parameters {
  real<lower=U> y_cens[N_cens];
  real mu;
  real<lower=0> sigma;
}
model {
  y_obs ~ normal(mu, sigma);
  y_cens ~ normal(mu, sigma);
}
>>>>>>> master