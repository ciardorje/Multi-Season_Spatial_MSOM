# Pantanal Fires Occupancy Model
A Hierarchical Multi Species Occupancy Model for estimating the effects of wildfires on the occurence of mammal species in the Brazillian Pantanal. This is a multi-season model, which incorporates the potential influence of spatial autocorellation on species occurence via a Gaussian Process intercept.

The model is written using the 'nimble' language and MCMC sampler. NIMBLE uses the same syntax as BUGS and JAGS, which are just other MCMC samplers, but I have found NIMBLE to be much faster than the other two (for these models at least). There is another MCMC sampler called STAN that is even faster, but it cannot sample discrete parameters (parameters that can only hold values of 0 or 1, e.g., whether or not a species is present at a site). I'm particularly interested in the discrete parameters in my research, hence I like NIMBLE, and I think you most likely will be interested in these parameters too. Plus, NIMBLE has a nice community page where you can ask questions https://groups.google.com/g/nimble-users

Here I provide the model script with detailed annotations. Then, in the fireHMSOM.R file, I provide just the model script itself and additionally include the code for the actual running of the model. 

## The Data Inputs ##

NIMBLE model inputs come under two categories: 

* ```Constants``` - Values that tend to just be used for indexing within the model structure. These values cannot be changed by the model and won't themselves be analysed.
* ```Data``` - The variables that will be analysed, i.e., the response and predictor variables. In our case we supply two main inputs       

<br>
For our model the constants and data will be:

* Constants:
  * ```nSps``` = The number of species you are modelling
  * ```nSites``` = The number of sites you sampled
  * ```nYears``` = The number of years you sampled for. If this varies among sites, this may need to be a vector of length ```nSites```, where values indicate the number of years each site was sampled for
  * ```nTrapWeeks``` = The number of independent sampling weeks (or whichever unit of time you group the photos by). If this is the same at every site across all years it could just be a single number. But if the length of time you deployed the camera traps varies between sites and years, it may need to be a matrix, with sites as rows, years as columns and cell values representing the number of weeks each site was sampled for in each year. 
  * ```nOCV``` = The number of predictor variables to be included in the occurence model
  * ```nDCV``` = The number of predictor variables to be included in the detection model

<br>

* Data:
  * ```Z``` = A binary 3D array, with dimensions ```nSites x nSps x nYears```, where cell values indicate whether (1) or not (0) each species was detected at each site in each year, across all trapping weeks combined.  
  * ```Y``` = A 4D binary array, with dimensions ```nSites x nSps x nYears x max(nTrapWeeks)```, where cell values indicate whether (1) or not (0) each species was detected at each site, in each trap week, in each year. 
    * If the number of trap weeks varies among sites and/or years, the length of the 4th dimension of the array should be the maximum number of trap weeks at any site in a single year; where sites were not sampled for the maximum number of trap weeks you can fill the additional cells with ```NA```, as the model indexing will mean that these cells are never accessed.
    * For example, if one site ```j``` was sampled for 20 weeks, the corresponding value in the ```nTrapWeeks``` variable provided in the constants will be 20. Therefore, even if the maximum number of sampling weeks at a single site was 30 weeks (and the 4th dimension of ```Y``` was thus of length 30), the model will only ever access up to the 20th cell in the row of Y corresponding to site ```j```.
  * ```SiteOCV``` = A 3D array, with dimensions ```nSites x nOCV x nYears```, holding the values of the occurrence model covariates (that is, variables you think may effect species occurence). Sites will be in rows and each covariate will be in a seperate column, and this structure will be repeated for each year (i.e., the third dimension)
  * ```SiteDCV``` = A 3D array, with dimensions ```nSites x nOCV x nYears```, holding the values of the detection model covariates (that is, variables you think may effect species detection). Same structure as ```SiteOCV```.
  * ```DMat2``` = A 2D matrix containing the squared geographic distance between each pair of camera trapping sites (both rows and columns will represent sites, and the cell values will represent the squared distance between a site in a given row and a site in a given column). This may need to be a 3D array if the position of camera traps varied with year, withe 3rd dimension representing years    

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
Occurrence probabilities for each species in each site will be linked to each other across years (i.e., temporal correlation), but first we will define the model for year 1, the first year of sampling. We only want to run this model on data from the first year of sampling, so we specify this with ```if(year == 1){}```. 
<br>
A quick note: this model assumes that the effects of covariates (i.e., burning, logging, etc.) on each species are consistent across years. This is a fair assumption, and if we were to allow these effects to vary between years it would likely result in a very complex model with very uncertain results.
```r
        if(year == 1){
        
          logit(psi[sp,site,year]) <- abar[sp] + a[sp,site,year] + inprod(bOCV[sp,1:nOCV], PatchOCV[site,year,1:nOCV])
  ```
Above is our first occurence model. The parameters are:
* ```psi[sp,site,year]``` = The probability of a species occuring at a given site in a given year. We calculate this on the logit scale, i.e. ``` logit() ```. This is because probabilities are constrained to values between 0 and 1, which can cause problems when when trying to estimate the effect sizes (slopes) of predictor variables. Logit transforming the probabilities puts them on a continuous scale, i.e., they can range between -∞ and ∞, and we can then back transform to true probability values (0-1).
* ```abar[sp]``` = The standard model intercept parameter for a given species, this is species-specific but is always the same regardless of year or site.
* ```a[sp,site,year]``` = The spatial autocorrelation parameter. 
  * I have let this vary by year, as in my research ongoing deforestation may result in sites becoming further apart over time. However, if your camera traps were always placed at the same spot every year, then you might not need to calculate this independently for each year. 
  * In my opinion, this whole parameter/concept is questionable. It works on the assumption that species are more likely to occur in sites that are closer to other sites, but it doesn't take into account whether or not the species actually occurred in the other sites. 
  * There are ways to take into account whether or not species occurred in neighbouring sites. However, these methods are also questionable as the whole point of this model is that we don't know for sure if a species actually does occur at any of the sites (that is, unless we actually see them there)! 
  * To be honest, I found the concept of this process cool so I wanted to see if I could code it, but it might not be worth using. There are ways to determine whether or not including this parameter improves model performance, which I can help with.
* ```bOCV[sp,1:nOCV]``` = A 2D matrix (species by covariate) containing the slope (b or beta) parameters for each of our predictor variables (also known as covariates). These are calculated independently for each species. We can have as many predictors as we want, so to keep the model tidy I have included these parameters in a matrix with the same number of columns as predictor variables, i.e. ```1:nOCV```, where nOCV represents the number of predictors: number(n) of occurence(O) covariates(CV).
* ```siteOCV[j,y,1:nOCV]``` = A 3D matrix (sites by year by covariate) containing the values of the predictor variables at each site in each year. You can include whatever you want as predictor variables/covariates (e.g., burn status, average temperature, etc. - whatever variables you think might affect a species' occurrence probability).

```inprod()``` is simply a tidy way of multiplying matrices (i.e, each beta/slope estimate ```bOCV``` multiplied by its corresponding site-specific covariate value ```siteOCV```). This just multiplies each value in one matrix by the corresponding value in the other matrix, e.g., ```matrix1[1,1] * matrix2[1,1]``` and so on, and then sums all the results.     

<br>

In the first year of sampling we didn't have any prior knowledge of species occurence. However, from year 2 onwards we have some idea of whether or not a species occured at a site in the previous year. We can incorporate this knowledge in our models for years 2, 3 and 4. To do this we define a second occurence model, which is only executed if the data is from year 2, 3 or 4. We express this criterion using an ```} else {``` statement, which is linked to the ```if(year == 1)``` statement above, i.e., ```if(year == 1)``` perform the model above, ```else``` perform the model below.

```r
      } else {

        logit(psi[sp,site,year]) <- abar[sp] + a[sp,site,year] + inprod(bOCV[sp,1:nOCV], PatchOCV[site,year,1:nOCV]) + 
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

We use a Bernoulli trial (which can only output either 0 or 1) with the probability set to the occurence probability generated by the above models, the ```psi``` variable. A good way to understand a Bernoulli trial is to think of a coin toss, the probability of the coin landing on either side is equal, at 0.5 or 50%. If we had a bernoulli trial with a probability of 0.5 we'd have an equal chance of getting either 0 or 1. As the probability increases so too does the chance that the bernoulli trial will yield a value of 1.    

<br>

Remember one of our inputs at the start was ```Z``` (note the upper case)? Well ```z``` (lower case) will contain the models attempts to correct for imperfect detection in our observed species occurence records (i.e., the values in that array ```Z```).    

<br>

The elements of our observed occurence records ```Z``` inform the model. However, the level of information we can gain from our observation records varies depending on whether or not the species was observed. Where ```Z[sp,site,year] = 1```, we know for certain that the species was at that site in that year, and the model can use this to inform upon its estimates of the effects of covariates on species occurrence. In these cases, the model will 'try' to generate a high occurence probability for that site in that year ```psi[sp,site,year]``` so that when we apply the Bernoulli trial to ```psi[sp,site,year]``` we are more likely to get a value of 1, indicating that the species is present, and thus matching ```Z[sp,site,year]```.     

<br>

However, where ```Z[sp,site,year] = 0``` this cannot be taken as a definitive indication that the species did not occur at that site in that year. We may just not have been able to detect it. So, for the sake of simplcity, the model will ignore these values. This isn't entirely true, as these non-observations are used to estimate detection probabilities in the detection models below, but in general little information can be gained from a non-observation (i.e., ```Z[sp,site,year] = 0```) compared to confirmed observations.


We provide the data to the model in the form of lists:
```r
mod_constants <- list()
mod_data <- list()
```






