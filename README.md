# Pantanal Fires Occupancy Model
A Hierarchical Multi Species Occupancy Model for estimating the effects of wildfires on the occurence of mammal species in the Brazillian Pantanal. This is a multi-season model, which incorporates the potential influence of spatial autocorellation on species occurence via a Gaussian Process intercept.

The model is written using the 'nimble' language and MCMC sampler. To put it (over)simply, MCMC sampling is just a method of estimating the possible values that statistical parameters of interest could take - this is done by iteratively drawing estimates from a potential distribution of parameter values, and we thus tend to run Bayesian models at least several thousand times. NIMBLE uses the same syntax as BUGS and JAGS, which are just other MCMC samplers, but I have found NIMBLE to be much faster than the other two (for these models at least). There is another MCMC sampler called STAN that is even faster, but it cannot sample discrete parameters (parameters that can only hold values of 0 or 1, e.g., whether or not a species is present at a site). I'm particularly interested in the discrete parameters in my research, hence I like NIMBLE, and I think you most likely will be interested in these parameters too. Plus, NIMBLE has a nice community page where you can ask questions https://groups.google.com/g/nimble-users

Here I provide the model script with detailed annotation. Then, in the fireHMSOM.R file I provide just the model script itself. 

## The Data Inputs ##

NIMBLE model inputs come under two categories: 

* ```constants``` - Values that tend to just be used for indexing within the model structure. These values cannot be changed by the model and won't themselves be modelled.
* ```data``` - The variables that will be modelled, i.e., the response and predictor variables.      

<br>
For our model the constants and data will be:

* Constants:
  * ```nSps``` = The number of species you are modelling.
  * ```nSites``` = The number of sites you sampled.
  * ```nYears``` = The number of years you sampled for. If this varies among sites, this may need to be a vector of length ```nSites```, where values indicate the number of years each site was sampled for.
  * ```nTrapWeeks``` = The number of independent sampling weeks (or whichever unit of time you group the photos by). If this is the same at every site across all years it could just be a single number. But if the length of time you deployed the camera traps varies between sites and years, it may need to be a matrix, with sites as rows, years as columns and cell values representing the number of weeks each site was sampled for in each year. 
  * ```nOCV``` = The number of predictor variables to be included in the occurence model/s.
  * ```nDCV``` = The number of predictor variables to be included in the detection model.

<br>

* Data:
  * ```Z``` = Our occurence records. A binary 3D array, with dimensions ```nSites x nSps x nYears```, where cell values indicate whether (1) or not (0) each species was detected at each site in each year, across all trapping weeks combined.  
  * ```Y``` = Our detection records. A 4D binary array, with dimensions ```nSites x nSps x nYears x max(nTrapWeeks)```, where cell values indicate whether (1) or not (0) each species was detected at each site, in each trap week, in each year. 
    * If the number of trap weeks varies among sites and/or years, the length of the 4th dimension of the array should be the maximum number of trap weeks at any site in a single year; where sites were not sampled for the maximum number of trap weeks you can fill the additional cells with ```NA```, as the model indexing will mean that these cells are never accessed.
    * For example, if one site ```j``` was sampled for 20 weeks, the corresponding value in the ```nTrapWeeks``` variable provided in the constants will be 20. Therefore, even if the maximum number of sampling weeks at a single site was 30 weeks (and the 4th dimension of ```Y``` was thus of length 30), the model will only ever access up to the 20th cell in the row of ```Y``` corresponding to site ```j```.
  * ```SiteOCV``` = A 3D array, with dimensions ```nSites x nOCV x nYears```, holding the values of the occurrence model covariates (that is, variables you think may effect species occurence). Sites will be in rows and each covariate will be in a seperate column, and this structure will be repeated for each year (i.e., the third dimension).
  * ```SiteDCV``` = A 3D array, with dimensions ```nSites x nOCV x nYears```, holding the values of the detection model covariates (that is, variables you think may effect species detection). Same structure as ```SiteOCV```.
  * ```DMat2``` = A 2D matrix containing the squared geographic distance between each pair of camera trapping sites (both rows and columns will represent sites, and the cell values will represent the squared distance between a site in a given row and a site in a given column). 
    * This may need to be a 3D array if the position of camera traps varied with year, where the 3rd dimension will represent years.    

In our case the two main inputs are ```Z``` and ```Y```, containing our observed occurences and detections, respectively. These two inputs will be used by the model to estimate occurence and detection probabilities, and the model will output a new version of ```Z``` containing it's estimates of species occurence, '*corrected*' for imperfect detection.     

We supply the model with the ```data``` and ```constants``` in the form of lists:
```r
data <- list(z = Z, y = Y, SiteOCV = SiteOCV, SiteDCV = SiteDCV, DMat2 = DMat2)
constants <- list(nSps = nSps, nSites = nSites, nYears = nYears, nTrapWeeks = nTrapWeeks, nOCV = nOCV, nDCV = nDCV)
```

## The Model ##

Now we have the inputs, we can create the model. First, we want to give the model a name ```fireMod``` and tell R that we are going to be providing some ```nimble``` code. This line does just that:
```r
fireMod <- nimbleCode({    
```    
### Likelihood ###
In the next section we define the model likelihood, i.e. the effects of our predictor variables on our outcomes of interest, that is species occurence and detection probabilities. There will be some parameters in this section that you will not see in the above list of data inputs. That's because these are the parameters we are interested in estimating. We will create space to 'place' these estimates later in the model, but for now I explain what each new parameter is after it is first mentioned.      


We estimate occurrence and detection parameters for every species in the data set individually. So we want to execute the model ```for``` each species seperately:
```r
  for(sp in 1:nSps){ 
```
We also estimate species occurrence and detection probabilities at each site independently. So we tell the model to do this too:
```r
    for(site in 1:nSites){ 
```
And, finally, we estimate occurrence and detection probabilities at each site in each year independently:
```r
      for(year in 1:nYears){  
```      

---

### **Occurence Model/s** ###

Occurrence probabilities for each species in each site will be linked to each other across years (i.e., temporal correlation), but first we will define the model for year 1, the first year of sampling. We only want to run this model on data from the first year of sampling, so we specify this with ```if(year == 1){}```. 
```r
        if(year == 1){
        
          logit(psi[sp,site,year]) <- abar[sp] + a[sp,site,year] + inprod(bOCV[sp,1:nOCV], PatchOCV[site,year,1:nOCV])
  ```
Above is our first occurence model, that for year 1. The parameters are:
* ```psi[sp,site,year]``` = The probability of a species occuring at a given site in a given year. 
  * We calculate this on the logit scale, i.e. ``` logit() ```. This is because probabilities are constrained to values between 0 and 1, which can cause problems when when trying to estimate the effect sizes (slopes) of predictor variables. 
  * Logit transforming puts the probabilities on a continuous scale, i.e., they can range between -∞ and ∞, and we can then back transform to true probability values (0-1).
* ```abar[sp]``` = The standard model intercept parameter for a given species, this is species-specific but is always the same regardless of year or site.
* ```a[sp,site,year]``` = The spatial autocorrelation parameter. This works on the assumption that species are more likely to occur in sites that are closer to other sites. So for sites close to other sites, this value will be higher. 
  * I have let this vary by year, as in my research ongoing deforestation may result in sites becoming further apart over time. However, if your camera traps were always placed at the same spot every year, then you might not need to calculate this independently for each year. 
  * In my opinion, this whole parameter/concept is questionable. It doesn't take into account whether or not the species actually occurred in the neighbouring sites.
  * There are ways to take into account whether or not species occurred in neighbouring sites. However, these methods are also questionable as the whole point of this model is that we don't know for sure if a species actually does occur at any of the sites (that is, unless we actually see them there)! 
  * To be honest, I found the concept of this process cool so I wanted to see if I could code it, but it might not be worth using. There are ways to determine whether or not including this parameter improves model performance, which I can help with.
* ```bOCV[sp,1:nOCV]``` = A 2D matrix (species by covariate) containing the slope (b or beta) parameters for each of our predictor variables (also known as covariates).
  * These are calculated independently for each species. 
  * We can have as many predictors as we want, so to keep the model tidy I have included these parameters in a matrix with the same number of columns as predictor variables, i.e. ```1:nOCV```, where nOCV represents the number of predictors: number(n) of occurence(O) covariates(CV).
  * A quick note: this model assumes that the effects of covariates (i.e., burning, logging, etc.) on each species are consistent across years. This is a fair assumption, and if we were to allow these effects to vary between years it would likely result in a very complex model with very uncertain results.     
* ```siteOCV[j,y,1:nOCV]``` = A 3D matrix (sites by year by covariate) containing the values of the predictor variables at each site in each year. 
  * You can include whatever you want as predictor variables/covariates (e.g., burn status, average temperature, etc. - whatever variables you think might affect a species' occurrence probability).

```inprod()``` is simply a tidy way of multiplying matrices (i.e, each beta/slope estimate ```bOCV``` multiplied by its corresponding site-specific covariate value ```siteOCV```). This just multiplies each value in one matrix by the corresponding value in the other matrix, e.g., ```bOCV[1,1,1] * siteOCV[1,1,1,1]``` and so on, and then sums all the results.     

<br>

In the first year of sampling we didn't have any prior knowledge of species occurence. However, from year 2 onwards we have some idea of whether or not a species occured at a site in the previous year. We can incorporate this knowledge in our models for years 2, 3 and 4. To do this we define a second occurence model, which is only executed if the data is from year 2, 3 or 4. We express this criterion using an ```} else {``` statement, which is linked to the ```if(year == 1)``` statement above, i.e., ```if(year == 1)``` perform the model above, ```else``` perform the model below.

```r
      } else {

        logit(psi[sp,site,year]) <- abar[sp] + a[sp,site,year] + inprod(bOCV[sp,1:nOCV], siteOCV[site,year,1:nOCV]) + 
                                    (theta[sp] * z[sp,site,(year-1)])
    
      }
```
The only difference here is the 'theta' parameter
* ```theta[sp]``` = The increase in occurence probability (on the logit scale) if the species DID occur within the site in the year before, i.e., ```(year-1)```. We know whether or not the species occurred in that site in the year before from ```z[sp,site,(year-1)]```...
* ```z[sp,site,(year-1)]``` = A binary variable indicating whether (1) or not (0) a species occured at a site in the year before. I'll cover more on how this is estimated in a moment. When a species didn't occur at a site in the year before ```z[sp,site,(year-1)]``` will be 0, and therefore ```(theta[sp] * z[sp,site,(year-1)])``` will also equal 0, and the theta parameter won't have any effect on occurence probability.     

This process (temporal correlation) could possibly be expanded to take into account multiple years before (i.e., if you had a sample from year 4 and you knew that the species occured at the site in all 3 previous years, the probability that the species would occur at the site in year 4 may be even higher than if it had just occurred there in year 3). I'd have to look into how to do this but I am happy to if you wanted to do that.    

<br>

That's the end of our occurence probability estimation. Now we want to know whether or not the species *actually* occured at each site (or at least estimate whether it did. These occurence estimates are represented by the ```z``` parameter mentioned above:

```r
    z[i,j,y] ~ dbern(psi[i,j,y])
```

Note the use of the ~ symbol. This tells the model this is a probabilistic node (a node is just a parameter that the model has been told to estimate). For probabilistic nodes we supply a probability density function on the right of the ~, and then the the model will randomly draw a value from the distribution and assign it to the object on the left of the ~.       

We use a Bernoulli trial (which can only output either 0 or 1) with the probability set to the occurence probability generated by the above models, the ```psi``` variable. A good way to understand a Bernoulli trial is to think of a coin toss, the probability of the coin landing on either side is equal, at 0.5 or 50%. If we had a bernoulli trial with a probability of 0.5 we'd have an equal chance of getting either 0 or 1. As the probability increases so too does the chance that the bernoulli trial will yield a value of 1.    

<br>

Remember one of our inputs at the start was ```Z``` (note the upper case)? Well ```z``` (lower case, the object in the above code) will contain the models attempts to correct for imperfect detection in our observed species occurence records (i.e., the values in that array ```Z```). In reality ```Z``` and ```z``` are the same object (we even specified that in our data input: ```data <- list(z = Z...```), but you can think of ```Z``` as an unchangeable version that will only *ever* include our observed occurences, while ```z``` can be altered by the model to add in instances (1 values) where the species occured but was not detected by our sampling.    

<br>

The elements of our observed occurence records ```Z``` inform the model. However, the level of information we can gain from our observation records varies depending on whether or not the species was observed. Where ```Z[sp,site,year] = 1```, we know for certain that the species was at that site in that year, and the model can use this to inform upon its estimates of the effects of covariates on species occurrence. In these cases, the model will 'try' to generate a high occurence probability for the species in that site in that year ```psi[sp,site,year]``` so that when we apply the Bernoulli trial to ```psi[sp,site,year]``` we are more likely to get a value of 1, indicating that the species is present, and thus matching ```Z[sp,site,year]```.     

<br>

However, where ```Z[sp,site,year] = 0``` this cannot be taken as a definitive indication that the species did not occur at that site in that year. We may just not have been able to detect it. So, for the sake of simplcity, the model will ignore these values. This isn't entirely true, as these non-observations are indirectly used to estimate detection probabilities in the detection models below (via our ```Y``` data input), but in general little information can be gained from a non-observation (```Z[sp,site,year] = 0```) compared to confirmed observations (```Z[sp,site,year] = 1```).

---

### **Detection Model** ###

So, we have now finished specifying our models of species occurence. But how can the model possibly know whether a species occured and was not detected (referred to as sampling zeros) or whether the species was not detected because it truly was not present (referred to as structural zeros)? This is done by seperate, but linked, models of species detection probabilities. We will define these now.     


Occupancy models depend on having repeat samples of the same site - this is because, for example, in cases where we detected a species at a site in one trap week ```Y[sp,site,year,week = 1] = 1```, but then didn't detect that species at the site in the next trap week ```Y[sp,site,year,week = 2] = 0```, the model can be confident that the non-observation in the second week represents a sampling zero rather than a structural zero, i.e., the species was there, we just didn't detect it. Based on this information alone, we could say the detection prbability was 0.5, as across two attempts we detected the species on one occasion (i.e., 50% of our samples). It is a little bit more complex though, as the model also incorporates knowledge of species detection across all sampled sites, as well as the potential effects of covariates on detection, specified in the detection model (below).    

Here's our detection model. This time it's the same across all years:
```r
    for(week in 1:nTrapWeeks[site,year]){  
          
      logit(p[sp,site,week,year]) <- lp[sp,year] + inprod(bDCV[sp,1:nDCV], siteDCV[site,week,year,1:nDCV])
      y[sp,site,week,year] ~ dbern(p[sp,site,week,year] * z[sp,site,year]) 

    }
  }
}
```

The parameters:
* ```p[sp,site,week,year]``` = The probability of detecting a given species in a given site in a given week in a given year. Again, this is on the logit scale as it is a probability being calaculated as the outcome of the linear effects of some predictors. 
* ```lp[sp,year]``` = This is the detection probability intercept. Here I have let it vary between years, as perhaps in one year the species declined in abundance throughout the landscape for some reason, making it harder to detect. To be honest though, I think it would be ok to just use the same value for all years and this would probably improve the certainty of our parameter estimates.
* ```bDCV[sp,1:nDCV]``` = A 2D matrix containing the slope parameters for all variables we may think influence the detection probability of a given species. I think camera trappers sometimes include camera trap ID as a random effect in detection models, in case some camera traps are slightly broken or others are just more sensitive for some reason.  
* ```y[sp,site,week,year]``` = This is the model's estimate of whether or not we detected a given species at a given site in a given week in a given year. Again, the model will 'try' to calculate values of ```bDCV``` that best replicate our observed detection matrix ```Y```, but in this case it will not only try and fit the parameters to the instances we did detect the species ```Y[sp,site,year,week] = 1``` but also those where we didn't detect the species ```Y[sp,site,year,week] = 0```, as we can be certain in both cases that the values we provide are true. We are not considering whether or not the species occured there, just whether our camera traps took a photo of it.     


Note that in our bernoulli trial to calculate ```y[sp,site,week,year]``` we set the probability to ```p[sp,site,week,year] * z[sp,site,year]```. This is because if a species didn't occur in a site (i.e., ```z[sp,site,year] = 0```) the probability that we will detect the species would always be 0 - te species isn't there so we are never going to detect it. In this way, detection is conditional on occurence, and the occurence and detection probability models are linked. The models are also conceptually linked in less direct ways:
* If a species was not observed by our camera traps, but it had a really high occurence probability ```psi``` and low detection probability ```p```, in all likelihood the species did occur at that site, but we just weren't able to detect it.
* Conversly, if a species wasn't observed at a site, had a really high detection probability ```p``` and a really low occurence probability ```psi``` then we could be fairly confident that the species was not detected because it truly was not present in that site.       

That is the end of the model likelihood section, now we move onto priors.     

---

### **Priors** ###

Bayesian models generate parameter estimates based on two things:
* Observed data (in our case, the arrays ```Z``` and ```Y```) 
* Prior knowledge we have about what values the parameters of interest could take     

We provide this prior knowledge in the form of probability distributions, from which the model draws possible parameter values. For example, if we had a variable we think could have a negative or a positive effect on a species' occurence, we would use a Normally distributed prior, as this can yield negative or positive values:

<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/92942535/222784865-db4e4575-956d-4c63-8642-5d090577db24.png">
</p>
<br/>

But if we had a variable that could only ever have a positive effect on species occurence (this is unlikely in reality, but is the easiest way to demonstrate the concept), we could use a half-normal distribution, where the probability of getting a negative value is always 0:

<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/92942535/222785175-c67819be-5c81-4074-9f58-bcb5b0b30e71.png">
</p>
<br/>

However, in most ecological situations we cannot be confident on what the parameter estimates will be before we run the model (i.e., we have limited *prior* knowledge). Therefore, we tend to use 'uninformative' or 'vague' priors - probability distributions with wide ranges. The model will draw values from these wide distributions and gradually converge on the distribution of values that the parameters being estimated are most likely to take, as it tailors the estimates to best reproduce our confirmed observations ```Z[] = 1``` and detection/non-detection data ```Y```.    

These narrowed down distributions are called posterior distributions and are what the model provides us as outputs. These posterior distributions are comparable with frequentist model parameter estimates, but instead of just having an estimate of the mean effect size (slope) and uncertainty around this (normally Standard Error), Bayesian models provide a probability distribution of all the values the parameter being estimated could take. We can use these posterior distributions to make inference on the effects of predictor variables , the occurence of species at sites, and so on. In fact, we can extract a posterior distribution for any parameter estimated in the model structure, which is why the possible analyses you could do using the outputs of these models are so exciting!   

Hierarchical Multi Species Occupancy models incorporate hierarchical structuring at the species and community level. Species level parameters are drawn from community level '*hyperparameter*' distributions - these are probability distributions that encompass all possible values a species-level parameter could take, across the entire community. As such, these hyperparameters will tend to have a wider range than the posterior distribution of the parameter estimates for any single species.   

For example, say we want to estimate the effect of the duration of fires on species-specific occurence ```BFire[1:nSps]```, where ```BFire``` is a ```1:nSp``` long vector containing estimates of the effect (slope) of fire duration on each species in the community. We typically use normally distributed priors to estimate slopes, and so the community-level hyperparameter will be represented by a mean ```BFire_sp_mean``` and standard deviation ```BFire_sp_sd``` - the parameters required to produce a Normal probability density function. We place prior distributions on these two parameters, from which potential parameter values will be drawn. Once these community-level distribution parameters have been estimated, the model will then draw an estimate of the slope for the effect of fire on each species ```BFire[1:nSps]``` from the resultant distribution, i.e.:
```r
#We set the priors for the community-level hyperparameter distribution
BFire_sp_mean ~ dnorm(0, sd = 1)  #We draw thee mean from a normal distribution, as  it could be positive OR negative
BFire_sp_sd ~ dunif(0, 5)   #Standard deviations can only be positive, so we use a uniform distribution 
                            #constrained to values between 0 and 5

#Then we draw species-specific fire responses from the community hyperparameter distribution
for(sp in 1:nSp){

  BFire[i] ~ dnorm(BFire_sp_mean, sd = BFire_sp_sd)  #We use a normal distribution, 
                                                     #inputting the estimated community hyperparameter mean and sd

}
```    
It is important to note that ```BFire[i] ~ dnorm(BFire_sp_mean, BFire_sp_sd)``` won't yield the same results as if we just drew 1000 random samples from the community hyperparameter distribution (i.e., ```rnorm(1000, BFire_sp_mean, BFire_sp_sd)```. This is due to the process used to estimate the parameters (MCMC sampling): remember, Bayesian models use prior knowledge we have about the values the parameter is likely to take (in this case ```dnorm(BFire_sp_mean, BFire_sp_sd)```) and observed data (```Z``` and ```Y```). The values of ```BFire[i]``` will represent a balance of the knowledge gained from these two sources of information, for each species.    

The advantage of this hierarchical structuring is that if we have species with very few detections ```i``` (and thus gain only limited information on where the species does and doesn't occur from our observed data), we can 'borrow' strength from the data for all of the other species in the community. This works on the rational that the community-level hyperparameters encompass all possible values that a species-level parameter could take across the entire community. So, although our data doesn't provide us with much information on the occurence of species ```i```, we can assume that the parameter value for species ```i``` lies somewhere within the community hypeerparameter distribution. Having said this, parameter estimates for species with very few observations tend to come with a greater degree of uncertainty, and we can thus be less confident in our estimates are than for a species with lots of observations. Nonetheless, these models can do a really good job of extracting reasonable parameter estimates from relatively sparse occurence data.     

The below figure visually demonstrates this concept: 
* The blue histogram represents the posterior distribution of possible ```BFire``` values across the entire community, i.e. ```dnorm(BFire_sp_mean, BFire_sp_sd)```
* The red histogram represents the posterior estimates of ```BFire``` for a species for which we have fairly good occurence records. Note that the variability of ```BFire``` for this species is quite low, i.e., we are pretty confident the true value resides somewhere around -0.6.
* The green histogram represents the posterior estimates of ```BFire``` for species ```i```, that imaginary species for which we have very few detections. Note, the distribution of possible ```BFire``` values for this species is a lot wider than for the species represented by the red histogram, as the model is less certain about what the true value of ```BFire``` is for species ```i```. However, as we told the model to draw values of ```BFire``` for species ```i``` from the community hyperparameter, the green histogram is still fully encompassed by the community hyperparameter distribution (in blue).  

<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/92942535/222967915-5198a5cd-83c3-4aea-b1c8-82dec5b5d942.png")>
</p>
<br/>

#### Community-Level Hyperparameters ####
Here are the community hyperparameters for our model. They all follow the same process as the above example, so I've just commented on what the priors specified on each line are referring to.    

You may notice, I haven't included community hyperparameters for the spatial autocorrelation intercept ```a```. This is mainly because the process used to calculate these values is computationally intensive (as you will probably gather from the species-level calculations we use for ```a``` in the next section) and I'm not sure the model would run/converge if we did incorporate hierarchical structuring in this parameter. I'd be happy to try though. 
 
```r

  abar.mean ~ dbeta(1, 1) #The mean of the community hyperparameter for the standard occurence model intercept on the probability scale
  mu.abar <- logit(abar.mean) #The mean of the community hyperparameter for the standard occurence model intercept on the logit scale
  sd.abar ~ dunif(0, 5) #The sd of the community hyperparameter for the standard occurence model intercept on the probability scale
    
  theta.mean ~ dnorm(0, 0.1) #The mean of the community hyperparameter for the temporal autocorrelation effect, on the probability scale
  mu.theta ~ logit(theta.mean) #The mean of the community hyperparameter for the temporal autocorrelation effect, on the logit scale
  sd.theta ~ dunif(0, 5) #The sd of the community hyperparameter for the temporal autocorrelation effect, on the logit scale
    
  for(year in 1:nYears){
      
    p.mean[year] ~ dbeta(1, 1)  #The mean of the community hyperparameter for the detection model intercept on the probability scale
    mu.lp[year] <- logit(p.mean[year]) #The mean of the community hyperparameter for the detection model intercept on the logit scale
    sd.lp[year] ~ dunif(0, 5)  #The sd of the community hyperparameter for the detection model intercept on the logit scale
      
    rho[year] ~ dunif(-1, 1)  #The coefficient of correlation between species occurence and detection probabilities
    sd.eta[year] <- sd.lp[year]/rho[year] #The covariance of the correllation between species occurrence and detection probabilities
      
  }
    
  for(n in 1:nDCV){
      
    mu.DCV[n] ~ dnorm(0, 0.1) #The mean of the community hyperparameter for the slope of detection covariate n
    sd.DCV[n] ~ dunif(0, 5) #The sd of the community hyperparameter for the slope of detection covariate n
      
  }
    
  for(n in 1:nOCV){
      
    mu.OCV[n] ~ dnorm(0, 0.1) #The mean of the community hyperparameter for the slope of occurrence covariate n
    sd.OCV[n] ~ dunif(0, 5) #The sd of the community hyperparameter for the slope of occurrence covariate n
      
  }
```

#### Species-Level Priors ####

Below are our species-level priors, all of which are a function of the community level hyperparameters, except the spatial autocorrelation parameter ```a```. Again these all follow the same structure as outlined in the example a couple sections up, so I've mostly just commented what each line refers to.    

The one thing which you may not recognise is the correlation between occurence and detection probabilities. This works on the rational that more common species (i.e., those with higher occurence probability ```psi```) tend to be easier to detect (and will thus have a higher detection probability ```p```). This is often a good reflection of reality and is widely accepted as a good thing to include. For full details of the parameters see Zipkin et al. 2009 (10.1111/j.1365-2664.2009.01664.x).

The spatial autocorrelation parameter is quite complex and would take a lot of words to explain - so I won't explain it, sorry. However, someone from my Masters cohort did a blog post about it if you want to know more (https://peter-stewart.github.io/blog/gaussian-process-occupancy-tutorial/). 

```r
   for(sp in 1:nSp){
   
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
      abar[sp] ~ dnorm(mu.abar, sd = sd.abar) #Intercept
      for(n in 1:nOCV){ bOCV[sp,n] ~ dnorm(mu.OCV[n], sd = sd.OCV[n]) } #Slopes for each occurence covariate n
      
      #Detection coefficients
      for(year in 1:nYears){
        
        #The mean of the prior for the species-level detection probability intercept is calculated based on a covariance matrix between detection and occurence
        #That is what the formula below does
        mu.eta[sp,year] <- mu.lp[year] + rho[year] * sd.lp[year]/sd.abar * (abar[sp] - mu.abar) 
        lp[sp,year] ~ dnorm(mu.eta[sp,year], sd = sd.eta[year]) #Prior for the species-specific detection model intercept on the logit scale 
        
      }
      
      for(n in 1:nDCV){ bDCV[sp,n] ~ dnorm(mu.DCV[n], sd = sd.DCV[n]) } #Slopes for each detection covariate n
      
    }
   }
```

---

### **Derived Quantities** ###

You can specify, in the model itself, any derived community metrics that you would like to calculate (e.g., species richness, beta diversity...), provided that they are a function of the estimated parameters (normally the ```z``` or ```psi``` matrices). For example, species richness at a given site in a given year would be calculated as ```Richness[site, year] <- sum(z[site,,year]```.       

However, I have found it to be much faster to calculate community metrics outside of the model itself, using the posterior distributions of ```z``` and ```psi```. By doing it this way you can also use external packages to work with the estimated parameters. For example, you could use the ```'betapart'``` package to calculate nestedness and turnover, whereas to do so within the model you would have to manually specify the calculations. I have previously written models including the calculations for nestedness and turnover, so if you wanted to calculate these values within the model, just to see how it works, I can add those in.
