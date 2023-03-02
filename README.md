# Pantanal Fires Occupancy Model
A Hierarchical Multi Species Occupancy Model for estimating the effects of wildfires on the occurence of mammal species in the Brazillian Pantanal. This is a multi-season model, which incorporates the potential influence of spatial autocorellation via a Gaussian Process intercept.

The model is written using the 'nimble' language and MCMC sampler. NIMBLE uses the same syntax as BUGS and JAGS, which are just other MCMC samplers, but I have found NIMBLE to be much faster than the other two (for these models at least). There is another MCMC sampler called STAN that is even faster, but it cannot sample discrete parameters (i.e., parameters that can only hold values of 0 or 1, e.g., whether or not a species is present at a site). I'm particularly interested in the discrete parameters in my research, hence I like NIMBLE, and I think you most likely will be interested in these parameters too. Plus, NIMBLE has a nice community page where you can ask questions https://groups.google.com/g/nimble-users

Here I provide the model script with detailed annotations. Then, in the fireHMSOM.R file, I provide just the model script itself and additionally include the code for the actual running of the model. 

## The Model ##

First, we want to give the model a name (fireMod) and tell R that we are going to be providing some NIMBLE code. This line does just that:
```r
fireMod <- nimbleCode({    
```
### Likelihood ###
In the next section we define the model likelihood, i.e. the effects of our predictor variables on our outcomes of interest, that is species occurence and detection probabilities. We estimate occurrence and detection parameters for every species i in the data set individually. So we want to execute the model ```for``` each species seperately:
```r
  for(sp in 1:nSps){ 
```
We also estimate species occurrence and detection parameters at each site independently. So we tell the model to do this too:
```r
    for(site in 1:nSites){ 
```
And, finally, we estimate occurrence and detection probabilities at each site in each year independently:
```r
      for(year in 1:nYears){  
```
Occurrence probabilities for each species in each site will be linked to each other across years (i.e., temporal correlation), but first we will define the model for year 1, the first year of sampling. We only want to run this model on data from the first year of sampling, so we specify this with ```if(year == 1){}```. 
<br>
A quick note: this model assumes that the effects of covariates (i.e., burning, logging, etc.) on each species are consistent across years. This is a fair assumption, and if we were to allow these effects to vary between years it would likely result in a very complex model with very uncertain results.
```r
        if(year == 1){
        
          logit(psi[sp,site,year]) <- abar[sp] + a[sp,site,year] + inprod(bOCV[sp,1:nOCV], PatchOCV[site,year,1:nOCV])
          
          }
```
Above is our first occurence model. The parameters are:
* ```psi[sp,site,year]``` = The probability of a species occuring at a given site in a given year. We calculate this on the logit scale, i.e. ``` logit() ```. This is because probabilities are constrained to values between 0 and 1, which can cause problems when when trying to estimate the effect sizes (slopes) of predictor variables. Logit transforming the probabilities puts them on a continuous scale, i.e., they can range between -∞ and ∞, and we can then back transform to true probability values (0-1).
* ```abar[sp]``` = The standard model intercept parameter for a given species, this is species-specific but is always the same regardless of year or site.
* ```a[sp,site,year]``` = The spatial autocorrelation parameter. I have let this vary by year, as in my research ongoing deforestation may result in sites becoming further apart over time. However, if your camera traps were always placed at the same spot every year, then you might not need to calculate this independently for each year. In my opinion, this whole parameter/concept is questionable. It works on the assumption that species are more likely to occur in sites that are closer to other sites, but it doesn't take into account whether or not the species actually occured in the other sites. There are ways to take into account whether species occur in neighbouring sites, but these methods are also questionable as the whole point of this model is that we don't know for sure if a species actually does occur at any of the sites (that is, unless we actually see them there)! To be honest, I found the concept of this process cool so I wanted to see if I could code it, but it might not be worth using. There are ways to determine whether or not including this parameter improves model performance, which I can help with.
* ```bOCV[sp,1:nOCV]``` = A 2D matrix (species by covariate) containing the slope (b or beta) parameters for each of our predictor variables (also known as covariates). These are calculated independently for each species. We can have as many predictors as we want, so to keep the model tidy I have included these parameters in a matrix with the same number of columns as predictor variables, i.e. ```1:nOCV```, where nOCV represents the number of predictors: number(n) of occurence(O) covariates(CV).
* ```siteOCV[j,y,1:nOCV]``` = A 3D matrix (sites by year by covariate) containing the values of the predictor variables at each site in each year, e.g., the mean temperature or percentage forest cover.

```inprod()``` is simply a tidy way of multiplying matrices (i.e, each beta/slope estimate ```bOCV``` multiplied by its corresponding site-specific covariate value ```siteOCV```). This ISN'T matrix multiplication, it just multiplies each value in one matrix by the corresponding value in the other matrix, e.g., ```matrix1[1,1] * matrix2[1,1]``` and so on, and then sums all the resultant values
<br>
You can include whatever you want as predictor variables/covariates (e.g., burn status, average temperature, etc. - whatever variables you think might effect a species' occurrence). You also don't necessarily have to include the same variables for each species, although we don't want to make the model too complex. For example, the presence of fruiting trees may influence frugivorous rodents, but there's little reason it would interest carnivorous jaguars (I know it probably indirectly would, as jaguars want to eat the rodents, but this was just the first example I could think of).
<br>
In the first year of sampling we didn't have any prior knowledge of species occurence. However, from year 2 onwards we have some idea of whether or not a given species occured at a given site the year before. We can incorporate this knowledge in our models for year 2, 3 and 4. To do this we define a second occurence model, which is only executed if the data is from year 2, 3 or 4. We tell nimble to only do this 
