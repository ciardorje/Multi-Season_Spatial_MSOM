# Pantanal Fires Occupancy Model
A Hierarchical Multi Species Occupancy Model for estimating the effects of wildfires on the occurence of mammal species in the Brazillian Pantanal. This is a multi-season model, which incorporates the potential influence of spatial autocorellation via a Gaussian Process intercept.

The model is written using the 'nimble' language and MCMC sampler. NIMBLE uses the same syntax as BUGS and JAGS, which are just other MCMC samplers, but I have found it to be much faster (for these models at least). There is another MCMC sampler called STAN that is even faster, but it cannot sample discrete parameters (i.e., parameters that can only be 0 or 1, like whether or not a species is present at a site). In my research I'm particularly interested in the discrete parameters, hence I like NIMBLE, I think you most likely will be too. Plus, NIMBLE has a nice community page where you can ask questions https://groups.google.com/g/nimble-users

Here I provide the model script with detailed annotations, and then in the fireHMSOM.R file I provide only the model script itself and additionally include the code for the actual running of the model. 

## The Model ##

First, we want to give the model a name (fireMod) and tell R that we are going to be providing some NIMBLE code. This line does just that.
```r
fireMod <- nimbleCode({    
```
### Likelihood ###
The section below defines the model likelihood, i.e. the effects of our predictor variables on our outcomes of interest: species occurence and detection probabilities. We estimate occurrence and detection parameters for every species i in the data set individually. So we want to perform the model 'for' each species i:
```r
  for(i in 1:nSps){ 
```
We also estimate species occurrence and detection parameters at each site independently. So we tell the model to do this too:
```r
    for(j in 1:nSites){ 
```
And, finally, we estimate occurrence and detection probabilities at each site in each year independently:
```r
      for(y in 1:nYears){  
```
Occurrence probabilities for each species in each site will be linked to each other across years, but first we will define the model for year 1, the first year of your sampling:
```r
        if(y == 1){
        
          logit(psi[i,j,y]) <- abar[i] + a[i,j,y] + inprod(bOCV[i,1:nOCV], PatchOCV[j,y,1:nOCV])
          
          }
```




                              #This works on the rational that if a species occurred at a site in year one, it is more likely to occur at that site in year two
                              #(assuming that the site hasn't been burnt, logged etc., but the model will account for temporal changes in these factors too)
                              
                              #N.B.: This model assumes that the effects of covariates (i.e., burning, logging, etc.) on each species are consistent across years
                              #This is a fair assumption, and if we did allow these effects to vary between years it would likely result in a very complex model with very uncertain results
  
  for(i in 1:nSps){         
    
    for(j in 1:nSites){       
      
      for(y in 1:nYears){     #We estimate occurrence and detection probabilities at each site in each year independently, but...
                              #occurrence probabilities for each species in each site will be linked to each other across years.
                              #This works on the rational that if a species occurred at a site in year one, it is more likely to occur at that site in year two
                              #(assuming that the site hasn't been burnt, logged etc., but the model will account for temporal changes in these factors too)
                              
                              #N.B.: This model assumes that the effects of covariates (i.e., burning, logging, etc.) on each species are consistent across years
                              #This is a fair assumption, and if we did allow these effects to vary between years it would likely result in a very complex model with very uncertain results
        
        if(y == 1){           #Here we define the occurence model for the first year of samping
                              #we symbolise 'year of sampling' as y, and if y == 1 then the code will perform the below model
          
          logit(psi[i,j,y]) <- abar[i] + a[i,j,y] + inprod(bOCV[i,1:nOCV], PatchOCV[j,y,1:nOCV])   #We don't have any knowledge of species occurrence before year 1, as sampling hadn't started 
                                                                                                   #Therefore, we just estimate species occurence probabilities as the outcome of our predictor variables 
                                                                                                   #(e.g., burn status, average temperature, etc. - whatever variables you think might effect a species' occurence can be included here) 
                                                                                                   #You also don't necessarily have to include the same variables for each species, although we don't want to make the model too complex 
                                                                                                   #(e.g., the presence of fruiting trees may have an effect on frugivorous rodents, but there's little reason it would interest carnivorous jaguars
                                                                                                   #[I know it probably indirectly would, as jaguars want to eat the rodents, but this was just the first example I could think of])
           #Above is our first occurence model, i.e., that for year 1
            #psi[i,j,y] = The probability of species i occuring at site j in year y. We calculate this on the logit scale as probabilities are constrained to values between 0 and 1
                          #and this is problematic when trying to estimate the effect sizes (slopes) of predictor variables. Logit transforming the probabilities puts them on a continuous scale
                          #i.e., they can range between -∞ and ∞, and we can then back transform to true probability values (0-1)
            #abar[i] = The standard model intercept parameter for species i, this is species-specific but is always the same regardless of year or site
            #a[i,j,y] = The spatial autocorrelation parameter. I have let this vary by year, as in my research ongoing deforestation may result in sites becoming further apart over time
                        #However, if your camera traps were always placed at the same spot every year, then it might not need to be calculated independently for each year
                        #In my opinion, this whole parameter/concept is questionable. 
                        #It works on the assumption that species are more likely to occur in sites that are closer to other sites, 
                        #but it doesn't take into account whether or not the species actually occured in the other sites.
                        #There are ways to take into account whether species occurr in neighbouring sites, but these methods are also questionable 
                        #as the whole point of this model is that we don't know for sure if a species actually does occur at any of the sites (unless we actually see them)!
                        #To be honest, I found the concept of this process cool so I wanted to see if I could code it, but it might not be worth using. 
                        #There are ways to determine whether or not including this parameter improves model performance, which I can help with.
            #bOCV[i,1:nOCV] = A 2D matrix (species by covariate) containing the slope (b or beta) parameters for each of our predictor variables. 
                              #These are calculated independently for each species i. 
                              #We can have as many predictors as we want, so to keep the model tidy I have included this as 1:nOCV,
                              #where nOCV represents the number of predictors [number(n) of occurence(O) covariates(CV)].
            #PatchOCV[j,y,1:nOCV] = A 3D matrix (Sites by year by covariate) containing the values of the predictor variables at site j in year y     
           
            #'inprod' is simply a tidy way of multiplying matrices (i.e, each beta estimate multiplied by its corresponding site-specific parameter value).
            #This ISN'T matrix multiplication it just multiplies each value in one matrix by the corresponding value in the other matrix,
            #e.g., matrix1[1,1] * matrix2[1,1] and so on, and then sums all the resultant values
          
          
        } else {      #If the data isn't from the first year of sampling then we perform this second occurence model instead
          
          #The only difference here is the 'theta' parameter
          #theta[i] = The increase in occurence probability (on the logit scale) if species i DID occur within site j in the year before (y-1)
          #This could possibly be expanded to take into account multiple years before y (i.e., for samples from the fourth year, if the species occured at the site in all 3 previous years,
          #the probability it would occur at the site in year 4 may be even higher). I'd have to look into how to do this but happy to if you wanted to do that. 
          
          logit(psi[i,j,y]) <- abar[i] + a[i,j,y] + inprod(bOCV[i,1:nOCV], PatchOCV[j,y,1:nOCV]) + (theta[i] * z[i,j,(y-1)])
          
        }
             
        z[i,j,y] ~ dbern(psi[i,j,y])    #This is where we determine whether (z[i,j,y] = 1) or not (z[i,j,y] = 0) species i occured in site j in year y
                                        #We use a Bernoulli trial (which can only output either 0 or 1) with the probability set to the occurence probability generated by the above models (psi)
                                        #Think of a coin toss, the probability of the coin landing on either side is equal, or 0.5 or 50%. 
                                        #If we had a bernoulli trial with a probability of 0.5 we'd have an equal chance of getting either 0 or 1 
                                        #As the probability increases so too does the chance that the bernoulli trial will yield a value of 1
                                        #One of our inputs to the model will be an observed occurence matrix Z (note capital Z). 
                                        #The elements of Z will indicate whether (Z[i,j,y] = 1) or not (Z[i,j,y] = 0) each species i was observed at site j in year y
                                        #Where Z[i,j,y] = 1 we know for certain that the species was there, and the model will use this to inform upon its estimates of the effect of covariates on species occurence
                                        #However, where Z[i,j,y] = 0, this cannot be taken as a definitive indication the species does not occur at the site, as we may just not have detected it
                                        #so for the sake of simplcity, the model will ignore these values (this isn't entirely true and is linked to the detection probability model below, 
                                        #but in general little information can be gained from a non-observation (i.e., Z[i,j,y] = 0) alone)
       
        #Now we move onto the detection probability model 
        
        #We model detection probabilities for each camera trap week w (or whichever unit of time you decide to use) in each site j in each year y
        for(w in 1:nTrapWeeks[j,y]){  
          
          logit(p[i,j,w,y]) <- lp[i,y] + inprod(bDCV[i,1:nDCV], SampleDCV[j,w,y,1:nDCV])    
          
          #p[i,j,w,y] = the probability of detecting species i in site j in week w in year y
          #lp[i,y] = This is the detection probability intercept for species i. Here I have let it vary between years, 
                     #as perhaps in one year the species declined in abundance throughout the landscape for some reason, making it harder to detect
                     #To be honest though, I think it would be ok to just use the same value for all years and this would probably improve the certainty of our parameter estimates
          #bDCV[i,1:nDCV] = A 2D matrix containing the slope parameters for all detection covariates for species i
          #SampleDCV[j,w,y,1:nDCV] = A 4D matrix containing the detection (D) covariate values (CV) for each trap week w at each site j in each year y
          
          y[i,j,w,y] ~ dbern(p[i,j,w,y] * z[i,j,y]) 
          
          #Here we estimate our observed data, the major input to our model, which will be a binary 4D array 'Y' 
          #(note capital Y, this is the data we observed, lower case y is the models' estimates of our observed data Y). 
          #This input array will indicate whether (Y[i,j,w,y] = 1) or not (Y[i,j,w,y] = 0) each species i was detected by our camera trap in site j in week w in year y
          #This (the array Y), and the instances where we did observe the species (i.e., Z[i,j,y] = 1), are the ONLY data we know for certain be true -
          #we can say with 100% certainty whether or not we took a photo of species i, 
          #but if we didn't take a photo of a secies, we cannot say for certain that the species was not in the area and just didn't happen to pass by the camera trap. 
          #Our model wll thus try to generate parameters that best replicate our observed data (capital Y) in its estimates of detections y
          #This presence/absence data (whether the species was there and not observed, or was just not there at all) is what we are trying to elucidate from our known data Y, 
          #and is represented in the z matrix we calculated using the occurence models above (i.e., z[i,j,y] ~ dbern(psi[i,j,y]) )
          
          #In this way, the detection and occurence models are linked - the species could only be detected if it was present at the site (i.e., z[i,j,y] = 1)
          #which is why when we estimate y[i,j,w,y] we use a bernoulli trial with probability (p[i,j,w,y] * z[i,j,y]). If the model suggests the species was not there 
          #z[i,j,y] = 0 and therefore the probability we give to the bernoulli trial will always be 0, as any value multiplied by 0 = 0 (i.e., you cannot detect a species if it is not there)
          #I don't think I can explain the inner workings, but basically:
            #1) if a species was not observed by our camera traps, but it had a really high occurence probability (psi) and low detection probability (p),
                #in all likelihood the species did occur at that site, but we just weren't able to detect it
            #2) Conversly, if a species wasn't observed at a site, had a really high detection probability (p) and a really low occurence probability (psi)
                #then we could be fairly confident that the species was not detected because it truly was not present in that site
          
        }
      }        #That is the end of the model likelihood section, now we move onto priors
      
      
      #####Priors#####
      
      #Bayesian models generate parameter estimates based on two things:
        #1) observed data (in our case the arrays Z and Y) 
        #2) prior knowledge we have about what values the parameters of interest could take
      #We provide this prior knowledge in the form of probability distributions, from which the model draws possible parameter values
        #e.g., if we had a variable we think could have a negative or a positive effect on a species' occurence we would use a Normally distributed prior, 
               #as this can yield negative or positive values 
               #But if we had a variable that could only ever have a positive effect on species occurence (this is quite unlikely in reality, but is the easiest way to demonstrate the concept), 
               #we could use a half-normal distribution, where the probability of getting a negative value is always 0
      #In most ecological situations we cannot be very confident on what the parameter estimate for a given species will be before we run the model
      #therefore, we tend to use 'uninformative' priors, probability distributions with very wide ranges
      #The model will draw values from this distribution and gradually converge on the true distribution of values that the parameter is most likely to take
      #These narrowed down distributions are called posterior distributions and are what the model provides us as outputs
      #We will then use these for making inference on the effects of predictor variables, the occurence of species at sites, etc.
      
      
      #####Species-Level Priors##### 
      
      #Spatial autocorrelation - Gaussian Process Intercept
      #This is the section for estimating the level of spatial autocorrelation in species occurence among sites
      #I don't think I can fully explain it here, and as I said before, it might not even be worth including
      #However, if you want to know more, someone from my Master's did a blog about it: https://peter-stewart.github.io/blog/gaussian-process-occupancy-tutorial/
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
      
      for(y in 1:nYears){ PatchR[j,y] <- sum(z[,j,y]) } #Calculates species richness at each site in each year
      
      #Beta diversity
      aEra <- sum(z[,j,1] * z[,j,2])
      bEra <- sum(z[,j,1] > z[,j,2])
      cEra <- sum(z[,j,2] > z[,j,1])
      
      #These formulae calculate 
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
