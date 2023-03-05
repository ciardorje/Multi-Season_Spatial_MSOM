fireMod <- nimbleCode({
  
  #####Likelihood#####
  
  for(sp in 1:nSpObs){
    for(site in 1:nSites){
      for(year in 1:nYears){
        if(year == 1){
          
          logit(psi[sp,site,year]) <- abar[sp] + a[sp,site,year] +  
                                        inprod(bOCV[sp,1:nOCV], PatchOCV[site,year,1:nOCV])
          
        } else {
          
          logit(psi[sp,site,year]) <- abar[sp] + a[sp,site,year] + 
                                        inprod(bOCV[sp,1:nOCV], PatchOCV[site,year,1:nOCV]) + 
                                          theta[sp] * z[sp,site,1]
          
        }
        
        z[sp,site,year] ~ dbern(psi[sp,site,year])
        
        for(week in 1:nTrapWeeks[site,year]){
          
          logit(p[sp,site,week,year]) <- lp[sp,year] + inprod(bDCV[sp,1:nDCV], SampleDCV[site,week,year,1:nDCV])    
          y[sp,site,week,year] ~ dbern(p[sp,site,week,year] * z[sp,site,year])
          
        }
      }
      
      
      #####Species-Level Priors##### 
      
      #Spatial autocorrelation - Gaussian Process Intercept
      alpha2[sp] ~ dexp(1)
      rho2[sp] ~ dexp(1)
      delta <- 0.01
      
      for(site in 1:nSites){
        for(year in 1:nYears){
          z_score[sp,site,year] ~ dnorm(0, sd = 1)
        }
      }
      
      for(year in 1:nYears){
        for(site in 1:(nSites - 1)){
          
          sigma[site,site,sp,year] <- alpha2[sp] + delta
          
          for(site2 in (site+1):nSites){
            
            sigma[site,site2,sp,year] <- alpha2[sp] * exp(-rho2[sp] * DMat2[site,l,year])
            sigma[site2,site,sp,year] <- sigma[site,site2,sp,year]
            
          }
        }
        
        sigma[nSites,nSites,sp,year] <- alpha2[sp] + delta
        sigma_cd[,,sp,year] <- chol(sigma[,,sp,year])
        a[sp,,year] <- sigma_cd[,,sp,year] %*% z_score[sp,,year]
        
      }
      
      #Temporal autocorrelation
      theta[sp] ~ dnorm(mu.theta, sd = sd.theta)
      
      #Occurence coefficients
      abar[sp] ~ dnorm(mu.abar, sd = sd.abar)
      for(n in 1:nOCV){ bOCV[sp,n] ~ dnorm(mu.OCV[n], sd = sd.OCV[n]) }
      
      #Detection coefficients
      for(year in 1:nYears){
        
        mu.eta[sp,year] <- mu.lp[year] + rho[year] * sd.lp[year]/sd.abar * (abar[sp] - mu.abar)
        lp[sp,year] ~ dnorm(mu.eta[sp,year], sd = sd.eta[year])
        
      }
      
      for(n in 1:nDCV){ bDCV[sp,n] ~ dnorm(mu.DCV[n], sd = sd.DCV[n]) }
      
    }
  }
    
  #####Community Hyperparameters#####
    
  abar.mean ~ dbeta(1, 1)
  mu.abar <- logit(abar.mean)
  sd.abar ~ dunif(0, 5)
    
  theta.mean ~ dnorm(0, 0.1)
  mu.theta ~ logit(theta.mean)
  sd.theta ~ dunif(0, 5)
    
  for(year in 1:nYears){
      
    p.mean[year] ~ dbeta(1, 1)
    mu.lp[year] <- logit(p.mean[year])
    sd.lp[year] ~ dunif(0, 5)
      
    rho[year] ~ dunif(-1, 1) 
    sd.eta[year] <- sd.lp[year]/rho[year]
      
  }
    
  for(n in 1:nDCV){
      
    mu.DCV[n] ~ dnorm(0, 0.1)
    sd.DCV[n] ~ dunif(0, 5)
      
  }
    
  for(n in 1:nOCV){
      
    mu.OCV[n] ~ dnorm(0, 0.1)
    sd.OCV[n] ~ dunif(0, 5)
      
  }
})
