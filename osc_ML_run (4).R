# Oscillator ML pred

# install packages and set working directory: If installed, then loading
require_packages = function(list_of_packages) {
  new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) install.packages(new_packages)
  lapply(list_of_packages, require, character.only = TRUE)
}

require_packages(c("elasticnet","randomForest"))

path = "/Users/danielpinelis/Downloads/" # INPUT
file_name = "Nasdaq vs model returns (5)" # INPUT

#### import data ### first column date second indepedent var third dependent var
raw_data = read.csv(paste(path,file_name,".txt",sep=""),sep="\t",header=TRUE,colClasses = c("character","numeric","numeric"))
raw_data[,c(2,3)] = raw_data[,c(2,3)]/100

#### fit, predict, hyperparamter tuning ###

# Gives training data based on year being predicted
#' @param yr character
#' 
#' @return dataframe subset of raw_data
get_yX_tr = function(yr) {
  return(raw_data[raw_data$Date<yr,-1][,c(2,1)])
}

# Gives independent variable observation for year
#' @param yr character
#' 
#' @return numeric
get_x_te = function(yr) {
  return(raw_data[raw_data$Date==yr,2])
}

# Fits one of three models and predicts one step ahead
#' @param yX_tr dataframe to train model
#' @param x_te independent observation for one-step ahead prediction
#' @param model_name character
#' @param model_params list
#' 
#' @return list(model,yX_tr,prediction)
fit_pred = function(yX_tr,x_te,model_name,model_params=NULL) {
  
  n_obs = nrow(yX_tr)
  
  if (model_name == "OLS") {
    yx_tr = data.frame(yX_tr)
    colnames(yX_tr) = c("target","x")
    fmodel = lm(target ~ x, yX_tr)
    x_te = data.frame(x_te)
    colnames(x_te) = "x"
    pred1 = predict(fmodel,x_te)
  }
  else if (model_name == "ElasticNet") {
    
    yx_tr = data.frame(yX_tr)
    colnames(yX_tr) = c("target","x")
    fmodel = lm(target ~ x, yX_tr)
    x_te = data.frame(x_te)
    colnames(x_te) = "x"
    pred1 = predict(fmodel,x_te)
    
    #fmodel = enet(matrix(yX_tr[,-1],nrow=n_obs),matrix(yX_tr[,1],nrow=n_obs),model_params$lambda)
    #alpha = en_model$penalty[which.min(abs(fmodel$penalty-model_params$lambda))]
    #alpha = max(0,alpha)
    #pred1 = predict(fmodel,x_tes=alpha,type="fit",mode="penalty")$fit
  }
  else if (model_name == "RandomForest") {
    fmodel = randomForest(matrix(yX_tr[,-1],nrow=n_obs),matrix(yX_tr[,1],nrow=n_obs), 
                          mtry=model_params$mtry, ntree=model_params$ntree, nodesize = model_params$node_perc*nrow(yX_tr), maxnodes = model_params$maxnodes)
    pred1 = predict(fmodel,newdata = x_te)
  }
  else {
    stop(paste("Invalid model name",model_name))
  }
  
  return(list(fmodel=fmodel,yX_tr=yX_tr,pred1=pred1))
}


# Computes out of sample R-squared 
#' @param pred numeric vector of predictions
#' @param act numeric vector of actual values
#' 
#' @return numeric R-squared
R_2 = function(pred,act) {
  1-sum((pred-act)^2)/sum(act^2)
}

# Computes out of sample R-squared 
#' @param pred numeric vector of predictions
#' @param act numeric vector of actual values
#' 
#' @return numeric R-squared
corr = function(pred,act) {
  cor(pred,act)
}

# Tunes model for best hyperparameters
#' @param model_name character
#' @param ev_metric function more positive values are treated as better
tune_model = function(model_name,ev_metric=R_2) {
  
  param_grids = list("RandomForest" = list(mtry =c(1),
                                           ntree=c(500),
                                           node_perc=seq(0,1,length.out=11),
                                           maxnodes=c(2,4,6,8,10)),
                     "ElasticNet" = list(lambdas = seq(0,1,length.out=11),
                                         alphas = seq(0,1,length.out=11)))
  
  
  param_combs = expand.grid(param_grids[[model_name]])
  n_comb = nrow(param_combs)
  params_ev_metric = rep(-Inf,n_comb)
  
  preds = matrix(NA,nrow=length(valid_yrs),ncol=n_comb) 
  
  for (i in 1:n_comb) {
    params_try = c(param_combs[i,])
    names(params_try) = names(param_grids[[model_name]])
    for (t in 1:length(valid_yrs)) {
      preds[t,i] = fit_pred(get_yX_tr(valid_yrs[t]),get_x_te(valid_yrs[t]),model_name,params_try)$pred1
    }
  }
  
  params_ev_metric = apply(preds,2,ev_metric,act=raw_data[raw_data$Date %in% valid_yrs,"Nasdaq"])
  best_ix = which(params_ev_metric == max(params_ev_metric))[1]
  opt_hyp_params = c(param_combs[best_ix,])
  names(opt_hyp_params) = names(param_grids[[model_name]])
  return(opt_hyp_params)
}


valid_start_yr = "1992"
start_yr = "1998" # INPUT first year of prediction with past data
indp_2022 = -.3 # INPUT independent variable for 2022

test_yrs = raw_data$Date[raw_data$Date>=start_yr] 
valid_yrs = raw_data$Date[raw_data$Date>=valid_start_yr & raw_data$Date<start_yr] 


# pick best model hyperparameters
model_oparams = list("OLS" = NULL,
                     "ElasticNet" = NULL,
                     "RandomForest"= NULL)
for (m in names(model_oparams)) {
  if (m != "OLS") {
    model_oparams[[m]] = tune_model(m)
  }
}


models_data = list("OLS" = list(preds=NULL,yrs_data=list()),
                   "ElasticNet" = list(preds=NULL,yrs_data=list()),
                   "RandomForest"=list(preds=NULL,yrs_data=list()))

for (t in 1:length(test_yrs)) {
  for (m in names(models_data)) {
    models_data[[m]]$yrs_data[[test_yrs[t]]] = fit_pred(get_yX_tr(test_yrs[t]),get_x_te(test_yrs[t]),m,model_oparams[[m]])
  }
}

# convert predictions to organized matrices
for (m in names(models_data)) {
  models_data[[m]][["preds"]]  = unlist(lapply(models_data[[m]][["yrs_data"]], function(x) x[["pred1"]]))
}

preds_2022 = c("OLS"=fit_pred(get_yX_tr("2022"),indp_2022,"OLS")$pred1,
               "Elastic Net"=fit_pred(get_yX_tr("2022"),indp_2022,"ElasticNet",model_oparams[["ElasticNet"]])$pred1,
               "Random Forest"=fit_pred(get_yX_tr("2022"),indp_2022,"RandomForest",model_oparams[["RandomForest"]])$pred1
               )

### portfolios ###

# Trading rule:
# Nasdaq weight = { 0, if MLpred <= 0
#                   1, else
#                 }

nasdaq_rets = raw_data[raw_data$Date>=start_yr,"Nasdaq"]
ols_rets = sign(models_data[["OLS"]][["preds"]])*nasdaq_rets
elastic_net_rets = sign(models_data[["ElasticNet"]][["preds"]])*nasdaq_rets
random_forest_rets = sign(models_data[["RandomForest"]][["preds"]])*nasdaq_rets

### results ###

annual_ret = function(pr) {
  return(mean(pr))
}

vol_ret = function(pr) {
  return(sd(pr))
}

sharpe = function(pr) {
  return(mean(pr)/sd(pr))
}

maxDD = function(pr) {
  cuml_rets = cumprod(1+pr)
  max_cuml_ret = cummax(c(1, cuml_rets))[-1]
  dds  = cuml_rets/max_cuml_ret - 1
  return(min(dds))
}

# Plots

# Equity curves
plot_dts = c(as.character(as.numeric(start_yr)-1),test_yrs)
u_ylim = max(cumprod(1+nasdaq_rets),cumprod(1+ols_rets),cumprod(1+elastic_net_rets),cumprod(1+random_forest_rets))
l_ylim = min(cumprod(1+nasdaq_rets),cumprod(1+ols_rets),cumprod(1+elastic_net_rets),cumprod(1+random_forest_rets))
plot(plot_dts,c(1,cumprod(1+nasdaq_rets)),main="Cumulative Returns for Models and S&P500 Index",xlab="",ylab="",type="l",log="y",ylim=c(l_ylim,u_ylim))
lines(plot_dts,c(1,cumprod(1+ols_rets)),col="red")
lines(plot_dts,c(1,cumprod(1+elastic_net_rets)),col="green")
lines(plot_dts,c(1,cumprod(1+random_forest_rets)),col="blue")
legend("topleft",lwd=2,legend=c("S&P500","OLS","ELastic Net","Random Forest"),
       cex = .7,bty = "n", col=c("black","red","green","blue"))

# Predictions vs actual returns
plot(test_yrs,nasdaq_rets,main="Predicted Returns for Models vs Actual Returns",xlab="",ylab="",type="l")
lines(test_yrs,models_data[["OLS"]][["preds"]],col="red")
lines(test_yrs,models_data[["ElasticNet"]][["preds"]],col="green")
lines(test_yrs,models_data[["RandomForest"]][["preds"]],col="blue")
legend("topright",lwd=2,legend=c("S&P500","OLS","ELastic Net","Random Forest"),
       cex = .7,bty = "n", col=c("black","red","green","blue"))


# Tables

# Statistical Results
results_M = matrix("",nrow=4,ncol=5)
rownames(results_M) = c("S&P500","OLS","Elastic Net","Random Forest")
colnames(results_M) = c("Annual Return","Volatility","Sharpe","MaxDD","R-squared")

results_M[1,] = c(round(c(annual_ret(nasdaq_rets)*100,vol_ret(nasdaq_rets)*100,sharpe(nasdaq_rets),maxDD(nasdaq_rets)*100),2),"")
results_M[2,] = round(c(annual_ret(ols_rets)*100,vol_ret(ols_rets)*100,sharpe(ols_rets),maxDD(ols_rets)*100,R_2(ols_rets,nasdaq_rets)),2)
results_M[3,] = round(c(annual_ret(elastic_net_rets)*100,vol_ret(elastic_net_rets)*100,sharpe(elastic_net_rets),maxDD(elastic_net_rets)*100,R_2(elastic_net_rets,nasdaq_rets)),2)
results_M[4,] = round(c(annual_ret(random_forest_rets)*100,vol_ret(random_forest_rets)*100,sharpe(random_forest_rets),maxDD(random_forest_rets)*100,R_2(random_forest_rets,nasdaq_rets)),2)

# output results table to csv
write.csv(results_M,paste(path,"results_M.csv",sep=""),row.names = TRUE)

# Pred vs Act returns
results_M2 = cbind(test_yrs,nasdaq_rets,models_data[["OLS"]][["preds"]],models_data[["ElasticNet"]][["preds"]],models_data[["RandomForest"]][["preds"]])
colnames(results_M2) = c("Date","S&P500","OLS","Elastic Net","Random Forest")
write.csv(results_M2,paste(path,"ActvsPred.csv",sep=""))

# Equity curves
results_M3 = cbind(plot_dts,c(1,cumprod(1+nasdaq_rets)),c(1,cumprod(1+ols_rets)),c(1,cumprod(1+elastic_net_rets)),c(1,cumprod(1+random_forest_rets)))
colnames(results_M3) = c("Date","S&P500","OLS","Elastic Net","Random Forest")
write.csv(results_M3,paste(path,"EquityC.csv",sep=""))


# prediction for 2022
print("Predictions for 2022 are:")
print(preds_2022)


