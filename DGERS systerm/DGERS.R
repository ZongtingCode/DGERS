### 联系方式: guzongtingoffice@gmail.com
data0 <- read.csv(file = "DGE-M14_17.csv",header = T)
data1 <- read.csv(file = "DGE-M14_19.csv",header = T)
#了解数据的格式
str(data0)
str(data1)
train.data <- na.omit(data0) #删除数据集data中的缺失数据
test.data <- na.omit(data1) 
##############
table(train.data$V46)
table(test.data$V46)
library(VIM) #缺失值的可视化
matrixplot(data0) #各变量分别看
matrixplot(data3)
#aggr(data) #整体可视化
aggr(train.data, numbers = TRUE)
aggr(test.data, numbers = TRUE)
aggr(data3, numbers = TRUE)
#1、首先告诉软件，哪些是你的分类变量。分类变量转为因子
for (i in names(train.data)[c(14:46)]) {train.data[,i] <- as.factor(train.data[,i])} # data[,"sex"] <- as.factor(data[,"sex"])
str(train.data)
for (i in names(test.data)[c(14:46)]) {test.data[,i] <- as.factor(test.data[,i])} # data[,"sex"] <- as.factor(data[,"sex"])
str(test.data)
table(train.data$V46) #logist回归模型每个变量的阳性结局样本数至少是其10倍（EPV）
table(test.data$V46) #（EPV，event per variable）估算纳入模型的最大变量数
##########
library(glmnet)
#2、因子的处理,分类变量处理,哑变量转换（one hot encoding，在data frame中定义ID）

x.factors <- model.matrix(train.data$V46 ~ train.data$V14+train.data$V15+train.data$V16+train.data$V17+train.data$V18+train.data$V19+train.data$V20
                          +train.data$V21+train.data$V22+train.data$V23+train.data$V24+train.data$V25+train.data$V26+train.data$V27+train.data$V28+train.data$V29+train.data$V30
                          +train.data$V31+train.data$V32+train.data$V33+train.data$V34+train.data$V35+train.data$V36+train.data$V37+train.data$V38+train.data$V39+train.data$V40
                          +train.data$V41+train.data$V42+train.data$V43+train.data$V44+train.data$V45)
##把连续变量捆绑起来##
x = as.matrix(data.frame(x.factors,train.data[,c(1:13)]))

#定义y
y=train.data[,46]
######-----Lasso纳入全部变量-----######
fit <- glmnet(x,y,family="binomial")
#解释偏差百分比以及相应的λ值
print(fit)
#解释偏差不再随着λ值的增加而减小，因此而停止
#画出收缩曲线图
plot(fit,label = FALSE)
plot(fit,label = TRUE,xvar = "lambda")#系数值如何随着λ的变化而变化
plot(fit,label = FALSE,xvar = "lambda")#删除变量标记
plot(fit,label = FALSE,xvar = "dev")#偏差与系数之间的关系图
#指定lamda给出相应的筛选变量
#lasso.coef <- predict(fit, type = "coefficients",s = 0.072010 ) # s指定
#lasso.coef

####---lasso cross-validation（内部交叉验证）获取最优λ值---####
#glmnet包在使用cv.glmnet()估计λ值时，默认使用10折交叉验证。在K折交叉验证中，
#数据被划分成k个相同的子集（折），#每次使用k-1个子集拟合模型，然后使用剩下的
#那个子集做测试集，最后将k次拟合的结果综合起来（一般取平均数），确定最后的参数。
#在这个方法中，每个子集只有一次用作测试集。在glmnet包中使用K折交叉验证非常容易，
#结果包括每次拟合的λ值和响应的MSE。默认设置为α=1。
#进行交叉验证，选择最优的惩罚系数lambada
##############################
set.seed(888)
cv.fit <- cv.glmnet(x,y,family="binomial")
plot(cv.fit) #画出收缩系数图（偏差）
cv.fit <- cv.glmnet(x,y,family="binomial",type.measure = "auc")#AUC和λ的关系
#cv.fit <- cv.glmnet(x,y,family="binomial")#不要反复运行
#画出收缩系数图（MSE）
cv.fit <- cv.glmnet(x,y,alpha = 1,family = "binomial",type.measure = "mse")
plot(cv.fit)
####提取变量（最小lambda，最佳）
cv.fit$lambda.min
coef(fit,s=cv.fit$lambda.min) #用cv.fit/fit模型筛选变量一致，s代表标准
#predict(cv.fit,type='coefficient',s=cv.fit$lambda.min) #功能同上
####提取变量（1倍标准误，最优）
cv.fit$lambda.1se
coef(fit,s=cv.fit$lambda.1se)
####注意：glmnet函数不能计算95%置信区间###
exp(coef(cv.fit)) #OR值
#predict(cv.fit,type='coefficient',s=cv.fit$lambda.1se) #功能同上
print(cv.fit)
#指定lamda给出相应的筛选变量
#predict(cv.fit,type='coefficient',s=0.1)
####把最佳和最优线加到收缩曲线图中####
plot(fit,xvar="lambda",label=FALSE)
abline(v=log(c(cv.fit$lambda.min,cv.fit$lambda.1se)),lty=2)
#导入验证集
#test <- read.csv(file = "test.csv",header = T)
#str(test)
####在test集当中设置新的x矩阵####
library(glmnet)
#for (i in names(test.data)[c(2:4,18,21,24:34,37:40)]) {test.data[,i] <- as.factor(test.data[,i])} # data[,"sex"] <- as.factor(data[,"sex"])
str(test.data)
#哑变量转换
newx.factors <- model.matrix(test.data$V46 ~ test.data$V14+test.data$V15+test.data$V16+test.data$V17+test.data$V18+test.data$V19+test.data$V20
                             +test.data$V21+test.data$V22+test.data$V23+test.data$V24+test.data$V25+test.data$V26+test.data$V27+test.data$V28+test.data$V29+test.data$V30
                             +test.data$V31+test.data$V32+test.data$V33+test.data$V34+test.data$V35+test.data$V36+test.data$V37+test.data$V38+test.data$V39+test.data$V40
                             +test.data$V41+test.data$V42+test.data$V43+test.data$V44+test.data$V45)
##把连续变量捆绑起来##
newx <- as.matrix(data.frame(newx.factors,test.data[,c(1:13)]))

#################利用lasso在test中预测值与真实值的比较#################
test.pred <- predict(cv.fit, newx = newx, type = "response", s=cv.fit$lambda.1se) #cv.fit为模型，response为返回概率值，S为惩罚函数
# <- predict(cv.fit, newx = newx, type = "response", s=cv.fit$lambda.min) #cv.fit为模型，response为返回概率值，S为惩罚函数
library(InformationValue)
misClassError(test.data$V46,test.pred)
plotROC(test.data$V46,test.pred)
library(pROC)
plot.roc(test.data$V46,test.pred,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE, of="thresholds",
         thresholds="best",
         print.thres="best")

#################利用lasso在train中预测值与真实值的比较#################
train.pred <- predict(cv.fit, newx = x, type = "response", s=cv.fit$lambda.1se)
library(InformationValue)
misClassError(train.data$V46,train.pred)
plotROC(train.data$V46,train.pred)
##########################################
#######建立logistic模型#################
######--Lasso筛选变量和全变量建立logistic模型--#####
#需要设置哑变量吗？不需要
lasso.fit <- glm(V46 ~ V13+V35+V36+V42, family = binomial, data = train.data)
####全变量建模，与lasso比较####
#full.fit <- glm(V46 ~ ., family = binomial, data = train.data)
full.fit <- glm(V46 ~ V4+V13+V14+V36+V38+V42, family = binomial, data = train.data) #"."代表剩余全部变量
####已发表文献建模，与lasso比较####
N1.fit <- glm(V46 ~ V15+V26+V39+V43, family = binomial, data = train.data)
#full.fit2 <- glm(class1 ~ u.size+u.shape+adhsn+s.size+nucl+chrom+n.nuc+mit, family = binomial, data = train)
summary(lasso.fit) #模型中的变量特征
exp(confint(lasso.fit)) #95%置信区间
exp(coef(lasso.fit)) #OR值
#其他模型#
summary(full.fit)
summary(N1.fit)
###----模型的比较----###
library(lmtest)
anova(lasso.fit,full.fit,test="Chisq") #模型有差异
AIC(lasso.fit,full.fit,N1.fit) #AIC越小，模型越优
lrtest(lasso.fit,full.fit) #Likelihood ratio test，模型相似性检验
lrtest(lasso.fit,N1.fit)
###----检查多重共线性VIF----###
library(car)
vif(lasso.fit) #方差膨胀因子vif小于5，提示无共线性问题
vif(full.fit)
###########
##qqPlot(full.fit)####QQ plot for studentized residuals not available for glm

#####----检查模型中变量间的相关性(最后做)----#####
library(corrplot)
for (i in names(train.data)[c(13,35,36,42)]) {train.data[,i] <- as.numeric(train.data[,i])} ##变量数值型转换
cr <- cor(train.data[,c(13,35,36,42)])
corrplot.mixed(cr)
corrplot(cr, method = "ellipse")
############

#---------------用lasso-logistic模型去训练集和验证集中预测--------------#
train.lasso.prob <- predict(lasso.fit, newdata = train.data, type = "response")
train.full.prob <- predict(full.fit, newdata = train.data, type = "response")
train.N1.prob <- predict(N1.fit, newdata = train.data, type = "response")
test.lasso.prob <- predict(lasso.fit, newdata = test.data, type = "response")
#其他模型#
test.full.prob <- predict(full.fit, newdata = test.data, type = "response")
test.N1.prob <- predict(N1.fit, newdata = test.data, type = "response")
#train.prob2[1:5] #inspect the first 5 predicted probabilities
#######----------预测校准度的检验（val函数）-------------#######
library(rms)
##---训练train集中---##
train.data$V46 <- as.numeric(train.data$V46) 
class(train.data$V46)
train.data$V46 <- ifelse(train.data$V46 == "1", 0, 1) # 转换为0和1
table(train.data$V46)
val.prob(train.lasso.prob,train.data$V46)
val.prob(train.full.prob,train.data$V46)
val.prob(train.N1.prob,train.data$V46)
#######---测试集test中---##
test.data$V46 <- as.numeric(test.data$V46) 
class(test.data$V46)
test.data$V46 <- ifelse(test.data$V46 == "1", 0, 1) # 转换为0和1
table(test.data$V46)
val.prob(test.lasso.prob,test.data$V46)
#其他模型#
val.prob(test.full.prob,test.data$V46)
val.prob(test.N1.prob,test.data$V46)
##----预测校准度的检验(Hosmer-Lemeshow拟合优度检验,适用于Logistic回归)----##
#install.packages("ResourceSelection")
library(ResourceSelection)
hoslem.test(test.data$V46,test.lasso.prob)

######   验证模型精确度、混淆矩阵、误判率   ##########
library(InformationValue)
####-------train中真实值与预测值的比较------####
#lasso.fit模型#
confusionMatrix(train.data$V46, train.lasso.prob)#混淆矩阵
optimalCutoff(train.data$V46, train.lasso.prob)#最优cutoff,模型在此时最优
misClassError(train.data$V46, train.lasso.prob)#误判率
plotROC(train.data$V46,train.lasso.prob) #左侧为实际发生，右侧为预测值
#方法2#
library(pROC)
roccurve <- (train.data$V46 ~ train.lasso.prob) #左侧为实际发生，右侧为预测值
plot.roc(roccurve, xlim = c(1,0),ylim = c(0,1))
auc(roccurve)
#方法3#
library(pROC)
plot.roc(train.data$V46,train.lasso.prob,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE, of="thresholds",
         thresholds="best",
         print.thres="best")
#full.fit模型#
confusionMatrix(train.data$V46, train.full.prob)#混淆矩阵
optimalCutoff(train.data$V46, train.full.prob)#最优cutoff,模型在此时最优
misClassError(train.data$V46, train.full.prob)#误判率
plotROC(train.data$V46,train.full.prob) #左侧为实际发生，右侧为预测值
library(pROC)
roccurve <- (train.data$V69 ~ train.full.prob) #左侧为实际发生，右侧为预测值
plot.roc(roccurve, xlim = c(1,0),ylim = c(0,1))
auc(roccurve)
library(pROC)
plot.roc(train.data$V46,train.full.prob,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE, of="thresholds",
         thresholds="best",
         print.thres="best")
#N1.fit模型#
confusionMatrix(train.data$V46, train.N1.prob)#混淆矩阵
optimalCutoff(train.data$V46, train.N1.prob)#最优cutoff,模型在此时最优
misClassError(train.data$V46, train.N1.prob)#误判率
plotROC(train.data$V46,train.N1.prob) #左侧为实际发生，右侧为预测值
library(pROC)
roccurve <- (train.data$V69 ~ train.N1.prob) #左侧为实际发生，右侧为预测值
plot.roc(roccurve, xlim = c(1,0),ylim = c(0,1))
auc(roccurve)
library(pROC)
plot.roc(train.data$V46,train.N1.prob,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE, of="thresholds",
         thresholds="best",
         print.thres="best")
####-------test中真实值与预测值的比较------####
#lasso.fit模型#
confusionMatrix(test.data$V46, test.lasso.prob)
misClassError(test.data$V46, test.lasso.prob)
plotROC(test.data$V46, test.lasso.prob) 
#方法2#
library(pROC)
roccurve <- (test.data$V46 ~ test.lasso.prob) #左侧为实际发生，右侧为预测值
plot.roc(roccurve, xlim = c(1,0),ylim = c(0,1))
auc(roccurve)
#方法3#
library(pROC)
plot.roc(test.data$V46,test.lasso.prob,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE, of="thresholds",
         thresholds="best",
         print.thres="best")
#full.fit模型#
confusionMatrix(test.data$V46, test.full.prob)
misClassError(test.data$V46, test.full.prob)
plotROC(test.data$V46, test.full.prob) 
library(pROC)
roccurve <- (test.data$V46 ~ test.full.prob) #左侧为实际发生，右侧为预测值
plot.roc(roccurve, xlim = c(1,0),ylim = c(0,1))
auc(roccurve)
library(pROC)
plot.roc(test.data$V46,test.full.prob,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE, of="thresholds",
         thresholds="best",
         print.thres="best")
#N1.fit模型#
confusionMatrix(test.data$V46, test.N1.prob)
misClassError(test.data$V46, test.N1.prob)
plotROC(test.data$V46, test.N1.prob) 
library(pROC)
roccurve <- (test.data$V46 ~ test.N1.prob) #左侧为实际发生，右侧为预测值
plot.roc(roccurve, xlim = c(1,0),ylim = c(0,1))
auc(roccurve)
library(pROC)
plot.roc(test.data$V46,test.N1.prob,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE, of="thresholds",
         thresholds="best",
         print.thres="best")
#######建立nomogram列线表模型#################
library(rms)
#rms包数据预处理--必须##
attach(train.data)
dd <- datadist(train.data)
options(datadist = "dd")
nom.fit <- lrm(V46 ~ V13+V35+V36+V42, data = train.data, x=T, y=T)
#经典模型的另一种表达形式#
nom.fit.full <- lrm(V46 ~ V4+V13+V14+V36+V38+V42, data = train.data, x=T, y=T)
nom.fit.N1 <- lrm(V46 ~ V15+V26+V39+V43, data = train.data, x=T, y=T)
##画出列线表##
nom.fit
nom <- nomogram(nom.fit, fun = plogis,fun.at = c(.001,.01,.05,seq(.1,.9, by = .1),.95,.99,.999), lp = F, funlabel = "DGE")
plot(nom)
###----提取Nonogram预测公式----###
#install.packages("nomogramEx")
library("nomogramEx")
nomogramEx(nomo=nom,np=1,digit = 9) #np代表列线表预测次数，digit代表小数点后的位数，默认为9

####----模型校准曲线(calibrate函数,仅用于lrm函数,理论上等同于前/下面的val函数)---####
library(rms)
cal <- calibrate(nom.fit, method = "boot", B = 100)
plot(cal, xlim = c(0,1.0), ylim = c(0,1.0) )
#经典模型#
library(rms)
cal <- calibrate(nom.fit.full, method = "boot", B = 100)
plot(cal, xlim = c(0,1.0), ylim = c(0,1.0) )
library(rms)
cal <- calibrate(nom.fit.N1, method = "boot", B = 100)
plot(cal, xlim = c(0,1.0), ylim = c(0,1.0) )
#######----------预测校准度的检验（val函数）注意：理论上glm与lrm拟合的logistic模型相同-------------#######
##---测试集test中---##
library(rms)
test.data$V46 <- as.numeric(test.data$V46) 
class(test.data$V46)
test.data$V46 <- ifelse(test.data$V46 == "1", 0, 1) # 转换为0和1
table(test.data$V46)
test.nom.prob <- predict(nom.fit, newdata = test.data, type = "fitted.ind") ##注意这里的type不同
plot(test.nom.prob)
val.prob(test.nom.prob,test.data$V46)
#经典模型#
test.nom.full.prob <- predict(nom.fit.full, newdata = test.data, type = "fitted.ind") ##注意这里的type不同
val.prob(test.nom.full.prob,test.data$V46)

test.nom.N1.prob <- predict(nom.fit.N1, newdata = test.data, type = "fitted.ind") ##注意这里的type不同
val.prob(test.nom.N1.prob,test.data$V46)
####----预测校准度的检验(Hosmer-Lemeshow拟合优度检验,适用于Logistic回归)----####
#install.packages("ResourceSelection")
library(ResourceSelection)
hoslem.test(test.data$V46,test.nom.prob)
hoslem.test(test.data$V46, test.lasso.prob)
hoslem.test(test.data$V46, test.full.prob)
hoslem.test(test.data$V46, test.N1.prob)
####----模型评价决策曲线分析(DCA，decision curve analysis)---####
#install.packages("rmda")
library(rmda)
# 注意：必须把结局变量转化为数值型，且为两分类“0”和“1”
train.data$V46 <- as.numeric(train.data$V46) 
class(train.data$V46)
train.data$V46 <- ifelse(train.data$V46 == "1", 0, 1) # 转换为0和1
table(train.data$V46)

mod.lasso <- decision_curve(V46 ~ V13+V35+V36+V42, 
                            family = binomial(link = 'logit'), data = train.data, thresholds = seq(0,1, by = 0.01),
                            confidence.intervals = 0.95, study.design = 'case-control', 
                            population.prevalence = 0.3) #结局发生率根据文献报道提供

mod.full <- decision_curve(V46 ~ V4+V13+V14+V36+V38+V42, 
                           family = binomial(link = 'logit'), data = train.data, thresholds = seq(0,1, by = 0.01),
                           confidence.intervals = 0.95, study.design = 'case-control', 
                           population.prevalence = 0.3) 

mod.N1 <- decision_curve(V46 ~ V15+V26+V39+V43, 
                           family = binomial(link = 'logit'), data = train.data, thresholds = seq(0,1, by = 0.01),
                           confidence.intervals = 0.95, study.design = 'case-control', 
                           population.prevalence = 0.3)

list <- list(mod.lasso,mod.full,mod.N1) #可以添加多个模型（>2）,用逗号隔开即可
##---绘制DCA曲线---##
plot_decision_curve(list, curve.names=c('lasso model','full model','N1 model'),
                    cost.benefit.axis = FALSE, col = c('red','blue','green'),
                    confidence.intervals = FALSE, standardize = FALSE) #也可以是‘green’
summary(mod.lasso,measure = "NB")
summary(mod.N1,measure = "NB")

##---分别绘制各模型临床影响曲线（预测风险数与真实风险数的比较）---##
plot_clinical_impact(mod.lasso, population.size = 1000, cost.benefit.axis = T,
                     n.cost.benefits = 8, col = c('red','blue'),
                     confidence.intervals = T, ylim = c(0,1000),
                     legend.position = "topright")

#plot_clinical_impact(mod.full, population.size = 1000, cost.benefit.axis = T,
#                     n.cost.benefits = 8, col = c('red','blue'),
#                     confidence.intervals = T, ylim = c(0,1000),
#                     legend.position = "topright")


#######----PERFECT ENDING----######

##################################################
