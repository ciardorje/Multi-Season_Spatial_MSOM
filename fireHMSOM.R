fireMod <- nimbleCode({
  
  #####Likelihood#####
  
  for(i in 1:nSpObs){
    for(j in 1:nSites){
      for(y in 1:nYears){
        
        if(y == 1){
          
          logit(psi[i,j,y]) <- abar[i] + a[i,j,y] + inprod(bOCV[i,1:nOCV], PatchOCV[j,y,1:nOCV])
          
        } else {
          
          logit(psi[i,j,y]) <- abar[i] + a[i,j,y] + inprod(bOCV[i,1:nOCV], PatchOCV[j,y,1:nOCV]) + theta[i] * z[i,j,(y-1)]
          
        }
        
        z[i,j,y] ~ dbern(psi[i,j,y])
        
        for(s in 1:nTrapWeeks[j,y]){
          
          logit(p[i,j,s,y]) <- lp[i,y] + inprod(bDCV[i,1:nDCV[y],y], SampleDCV[j,s,y,1:nDCV[y]])    
          y[i,j,s,y] ~ dbern(p[i,j,s,y] * z[i,j,y])
          
        }
      }
      
      
      #####Species-Level Priors##### 
      
      #Spatial autocorrelation - Gaussian Process Intercept
      alpha2[i] ~ dexp(1)
      rho2[i] ~ dexp(1)
      delta <- 0.01
      
      for(j in 1:nPatches){
        for(y in 1:nYears){
          z_score[i,j,y] ~ dnorm(0, sd = 1)
        }
      }
      
      for(y in 1:nYears){
        for(j in 1:(nPatches - 1)){
          
          sigma[j,j,i,y] <- alpha2[i] + delta
          
          for(l in (j+1):nPatches){
            
            sigma[j,l,i,y] <- alpha2[i] * exp(-rho2[i] * DMat2[j,l,y])
            sigma[l,j,i,y] <- sigma[j,l,i,y]
            
          }
        }
        
        sigma[nPatches,nPatches,i,y] <- alpha2[i] + delta
        sigma_cd[,,i,y] <- chol(sigma[,,i,y])
        a[i,,y] <- sigma_cd[,,i,y] %*% z_score[i,,y]
        
      }
      
      #Temporal autocorrelation
      theta[i] ~ dnorm(mu.theta, sd = sd.theta)
      
      #Occurence coefficients
      abar[i] ~ dnorm(mu.abar, sd = sd.abar)
      for(n in 1:nOCV){ bOCV[i,n] ~ dnorm(mu.OCV[n], sd = sd.OCV[n]) }
      
      #Detection coefficients
      #Integrated model - separate detection parameters for each sampling method 
      for(y in 1:nYears){
        
        mu.eta[i,y] <- mu.lp[y] + rho[y] * sd.lp[y]/sd.abar * (abar[i] - mu.abar)
        lp[i,y] ~ dnorm(mu.eta[i,y], sd = sd.eta[y])
        
        for(n in 1:nDCV[y]){ 
          
          bDCV[i,n,y] ~ dnorm(mu.DCV[n,y], sd = sd.DCV[n,y]) 
          
        }
      }
    }
    
    #####Community Hyperparameters#####
    
    abar.mean ~ dbeta(1, 1)
    mu.abar <- logit(abar.mean)
    sd.abar ~ dunif(0, 5)
    
    theta.mean ~ dnorm(0, 0.1)
    mu.theta ~ logit(theta.mean)
    sd.theta ~ dunif(0, 5)
    
    for(y in 1:nYears){
      
      p.mean[y] ~ dbeta(1, 1)
      mu.lp[y] <- logit(p.mean[y])
      sd.lp[y] ~ dunif(0, 5)
      
      rho[y] ~ dunif(-1, 1) 
      sd.eta[y] <- sd.lp[y]/rho[y]
      
      for(n in 1:nDCV[y]){
        
        mu.DCV[n,y] ~ dnorm(0, 0.1)
        sd.DCV[n,y] ~ dunif(0, 5)
        
      }
    }
    
    for(n in 1:nOCV){
      
      mu.OCV[n] ~ dnorm(0, 0.1)
      sd.OCV[n] ~ dunif(0, 5)
      
    }
    
    #####Derived Quantities#####
    
    for(j in 1:nPatches){
      
      for(y in 1:nYears){ PatchR[j,y] <- sum(z[,j,y]) }
      
      aEra <- sum(z[,j,1] * z[,j,2])
      bEra <- sum(z[,j,1] > z[,j,2])
      cEra <- sum(z[,j,2] > z[,j,1])
      
      EraSor[j] <- (bEra + cEra) / (2 *  + bEra + cEra)
      EraTurn[j] <- min(bEra, cEra) / (aEra + min(bEra, cEra))
      EraNest[j] <- EraSor[j] - EraTurn[j]
      
    }
    
    for(y in 1:nYears){
      for(j in 1:nPatches){
        
        PairSor[j,j,y] <- 0
        PairTurn[j,j,y] <- 0
        PairNest[j,j,y] <- 0
        
        for(l in (j+1):nPatches){
          
          aPair <- sum(z[,j,y] * z[,l,y])
          bPair <- sum(z[,j,y] > z[,l,y])
          cPair <- sum(z[,l,y] > z[,j,y])
          
          PairSor[j,l,y] <- (bPair + cPair) / (2 * aPair + bPair + cPair)
          PairTurn[j,l,y] <- min(bPair, cPair) / (aPair + min(bPair, cPair))
          PairNest[j,l,y] <- PairSor[j,l,y] - PairTurn[j,l,y]
          
          PairSor[l,j,y] <- PairSor[j,l,y]
          PairTurn[l,j,y] <- PairTurn[j,l,y]
          PairNest[l,j,y] <- PairTurn[j,l,y]
          
        }
      }
    }
    
    for(y in 1:nYears){ 
      for(i in 1:nSpObs){ 
        
        nOcc[i,y] <- sum(z[i,,y]) 
        
      }
    }
  }
})
