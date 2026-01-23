library(fda)
data("CanadianWeather")
names(CanadianWeather)

# Response variable we'll use throughout: log total annual precipitation.
annualprec = log10(apply(daily$precav, 2, sum))
length(annualprec)

# Create a 65-dimensional Fourier basis to use without a roughness penalty.
tempbasis = create.fourier.basis(c(0, 365), 65)
tempSmooth = smooth.basis(day.5, daily$tempav, tempbasis)
tempfd = tempSmooth$fd

# Store the two functional data covariates required for regression, intercept and functionals, in a list of length two.
templist = vector("list", 2)
templist[[1]] = rep(1, 35)
onebasis = create.constant.basis(c(0, 365))
onefd = fd(matrix(1, 1, 35), onebasis)
templist[[1]] = onefd
templist[[2]] = tempfd
templist

###############################
# Low-dimensional regression coefficient function beta
###############################

# We'll use 5 Fourier basis functions for beta, along with a constant function for alpha.
conbasis = create.constant.basis(c(0, 365))
betabasis = create.fourier.basis(c(0, 365), 5)
betalist = vector("list", 2)
betalist[[1]] = fdPar(conbasis)
betalist[[2]] = betabasis

# Now call the functional linear regression function.
fRegressList = fRegress(annualprec, templist, betalist)
names(fRegressList)
fRegressList$df # 6 degrees of freedom

# Plot the estimated regression function.
betaestlist = fRegressList$betaestlist
tempbetafd = betaestlist[[2]]$fd
plot(tempbetafd, xlab="Day", ylab="Beta for temperature")

# Intercept term.
coef(betaestlist[[1]])
# All coefficients.
coef(betaestlist[[2]])

# Assess the quality of the fit. First, compute residuals and error sums of squares.
annualprechat1 = fRegressList$yhatfdobj
annualprecres1 = annualprec - annualprechat1
SSE1.1 = sum(annualprecres1^2)
SSE0 = sum((annualprec - mean(annualprec))^2)

# Now, report R^2 and F-ratio.
RSQ1 = (SSE0 - SSE1.1)/SSE0 # 0.80
Fratio1 = ((SSE0 - SSE1.1)/5) / (SSE1.1/29) # 22.6 with 5 and 29 degrees of freedom.

###############################
# Coefficient beta estimate using a roughness penalty
###############################

# Set up a harmonic acceleration operator for penalization.
Lcoef = c(0, (2*pi/365)^2, 0)
harmaccelLfd = vec2Lfd(Lcoef, c(0, 365))

# Replace the previous beta basis with a functional parameter object that incorporates this roughness penalty and smoothing.
betabasis = create.fourier.basis(c(0, 365), 35)
lambda = 10^12.5
betafdPar = fdPar(betabasis, harmaccelLfd, lambda)
betalist[[2]] = betafdPar

# Use fRegress to estimate the regression coefficients and predicted values.
annPrecTemp = fRegress(annualprec, templist, betalist)
betaestlist2 = annPrecTemp$betaestlist
annualprechat2 = annPrecTemp$yhatfdobj
annPrecTemp$df # 4.7 degrees of freedom, a little lower than the 6 from before! Recall, this is the trace of the hat matrix

# Plot the estimated regression function.
tempbetafd2 = betaestlist2[[2]]$fd
plot(tempbetafd2, xlab="Day", ylab="Beta for temperature") # This matches the book!

# Report R^2 and F-ratio.
SSE1.2 = sum((annualprec-annualprechat2)^2)

# Now, report R^2 and F-ratio.
RSQ2 = (SSE0 - SSE1.2)/SSE0 # 0.75, a small drop compared to earlier. However, we use fewer DoF
Fratio2 = ((SSE0 - SSE1.2)/3.7) / (SSE1.2/30.3) # 25.1 with 3.7 and 30.3 degrees of freedom. Even more significant!

# Plot actual vs. predicted.
plot(annualprechat2, annualprec, xlab="Predicted", ylab="Actual")
lines(annualprechat2, annualprechat2, lty=2)

# Does our model actually improve the fit? What if we just use constant beta?
betalist[[2]] = fdPar(conbasis)
fRegressList = fRegress(annualprec, templist, betalist)
betaestlist = fRegressList$betaestlist
fRegressList$df # 2 degrees of freedom

# Rerport R^2 and F-ratio.
annualprechat = fRegressList$yhatfdobj
SSE1 = sum((annualprec-annualprechat)^2)
RSQ = (SSE0 - SSE1)/SSE0 # 0.49, much worse
Fratio = ((SSE0-SSE1)/1)/(SSE1/33) # 31.3 with 1 and 33 degrees of freedom. Still significant!

###############################
# Choosing smoothing parameters (DOESN'T WORK)
###############################

# We plot cross-validation scores for various values of lambda using the hat matrix.
loglam = seq(5, 15, 0.5)
nlam = length(loglam)
SSE.CV = matrix(0, nlam, 1)
for (ilam in 1:nlam){
  lambda = 10^loglam[ilam]
  betalisti = betalist
  betafdPar2 = betalisti[[2]]
  betafdPar2$lambda = lambda
  betalisti[[2]] = betafdPar2
  fRegi = fRegress.CV(annualprec, templist, betalisti)
  SSE.CV[ilam] = fRegi$SSE.CV
}
plot(loglam, SSE.CV)

###############################
# Confidence intervals (DOESN'T WORK)
###############################

resid = annualprec - annualprechat
SigmaE.= sum(resid^2)/(35-fRegressList$df)
SigmaE = SigmaE.*diag(rep(1,35))
y2cMap = tempSmooth$y2cMap
stderrList = fRegress.stderr(fRegressList, y2cMap,
                             SigmaE)

###############################
# Scalar response models by functional principal components (DOESN'T WORK)
###############################

daybasis365=create.fourier.basis(c(0, 365), 365)
lambda =1e6
tempfdPar =fdPar(daybasis365, harmaccelLfd, lambda)
tempfd =smooth.basis(day.5, daily$tempav,
                     tempfdPar)$fd

lambda = 1e0
tempfdPar = fdPar(daybasis365, harmaccelLfd, lambda)
temppca = pca.fd(tempfd, 4, tempfdPar)
harmonics = temppca$harmonics

pcamodel = lm(annualprec~temppca$scores)
pcacoefs = summary(pcamodel)$coef
betafd = pcacoefs[2,1]*harmonics[1] +
  pcacoefs[3,1]*harmonics[2] +
  pcacoefs[4,1]*harmonics[3]
coefvar = pcacoefs[,2]^2
betavar = coefvar[2]*harmonics[1]^2 +
  coefvar[3]*harmonics[2]^2 +
  coefvar[4]*harmonics[3]^2

plot(betafd, xlab="Day", ylab="Regression Coef.",
     ylim=c(-6e-4,1.2e-03), lwd=2)
lines(betafd+2*sqrt(betavar), lty=2, lwd=1)
lines(betafd-2*sqrt(betavar), lty=2, lwd=1)

###############################
# Statistical tests
###############################

# We test whether the result obtained from our observed data is different than if we permuted the response variable vector.
F.res = Fperm.fd(annualprec, templist, betalist)
F.res$Fobs # 3.03, as long as we use the Fourier basis and not the constant basis.
F.res$qval # Critical value is 0.3, so our result is statistically significant.
