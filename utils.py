
"""
Common functions used by the main programs
"""

import numpy as np
import math
import sys
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss,hinge_loss


def getHingeLoss(index,y_true,clf_scores,weights):
    '''
    returns the hinge loss of the ensemble for a pattern
     on the position given by index
    
     y_true-the array of scores
     clf_scores-the scores of all the classifieres
     weights-the weights of the classifiers
     
    '''
    
    M=len(clf_scores)
    
    sum1=0
    
    for i in range(M):
        sum1+=weights[i]*clf_scores[i][index]
    
    hl=max(0,1-y_true[index]*sum1)
    
    return hl
    

def getHingeLossEns(y_true,clf_scores,weights):
    
    '''
    returns the hinge loss of the ensemble 
    
     y_true-the array of scores
     clf_scores-the scores of all the classifieres
     weights-the weights of the classifiers
     
    '''
    
    N=len(y_true)  
    
    err=0
    
    for i in range(N):
        err+=getHingeLoss(i,y_true,clf_scores,weights)
    
    err=err/N
    
    return err


def ambiguity(h,y):
   '''Returns the ambiguity of the ensemble, see equation(5) 
      from the paper
       
      h-the matrix of predictions of all the classifiers
      y-the array of targets
   '''


   H=[np.sign(sum(h[:,j])) for j in range(len(h[0]))]

   a = sum(sum((H-h)*y)/(2*len(h)*len(y)))

   return a,H


def getErrorClf(pred_clf,y_n):
    '''
    Returns the error (0-1 loss) of one classifier
    
    pred_clf-the predictions of the classifiers
    y_n-the argets
    '''
    
    error=0
    
    for i in range(len(pred_clf)):
        if pred_clf[i]*y_n[i]==-1:
            error+=1
        elif pred_clf[i]*y_n[i]==0:
            error+=0.5
    
    N=len(y_n)

    error=error/N      

    return error

def getErrorsClfs(matr_pred,y_n):
    '''
    Returns the list of errors (0-1 loss) of all classifiers
    
    matr_pred-the matrix of predictions of all classifiers
    y_n-the argets
    '''
    
    err_clf=[]
    for i in range(len(matr_pred)):
        err_clf.append(getErrorClf(matr_pred[i],y_n))
        
    return err_clf

def getDivValuesCE(bagger,Xtr,ttr,Xte,tte,etest,etrain,atest,atrain):
    '''
    Calculates the train/test CE and the train/test amb_CE for the ensemble formed
    of 100 trees
    
    bagger-the bagging object
    Xtr-training data
    ttr-training targets
    Xte-test data
    tte-test targets
    etest-array of test CE used for different forests
    etrain-array of train CE sed for different forests
    atest-array of test amb_CE
    atrain-array of train amb_CE
    '''

    htr_proba,Htr_proba=getMatrPredCE(bagger.estimators_, Xtr, ttr)
    hte_proba,Hte_proba=getMatrPredCE(bagger.estimators_, Xte, tte)
    
    etrain.append(log_loss(ttr,Htr_proba))
    etest.append(log_loss(tte,Hte_proba))

    nr_clf=len(htr_proba)
    c=np.ones(nr_clf)*(1/nr_clf)
     
    div_tr=div_ens(ttr,htr_proba,c)
    div_tst=div_ens(tte,hte_proba,c)
     
    atrain.append(div_tr)
    atest.append(div_tst)
    

def getDivValuesCE5(trees_list,Xtr,ttr,Xte,tte,etest_5,etrain_5,atest_5,atrain_5):
    
     '''
    Calculates the train/test CE and the train/test amb_CE for the ensemble formed
    of 5 trees
    
    trees_list-list of 5 trees
    Xtr-training data
    ttr-training targets
    Xte-test data
    tte-test targets
    etest-array of test CE used for different forests
    etrain-array of train CE sed for different forests
    atest-array of test amb_CE
    atrain-array of train amb_CE   
    '''
    
    
    htr_proba,Htr_proba=getMatrPredCE(trees_list, Xtr, ttr)
    hte_proba,Hte_proba=getMatrPredCE(trees_list, Xte, tte)

     
    etrain_5.append(log_loss(ttr,Htr_proba))
    etest_5.append(log_loss(tte,Hte_proba))
  
    nr_clf=len(htr_proba)
    c=np.ones(nr_clf)*(1/nr_clf)
     
    div_tr=div_ens(ttr,htr_proba,c)
    div_tst=div_ens(tte,hte_proba,c)
     
    atrain_5.append(div_tr)
    atest_5.append(div_tst)


def getDivValuesHL(bagger,Xtr,ttr,Xte,tte,etest_HL,etrain_HL,atest_HL,atrain_HL):
    
    '''
    Calculates the train/test hinge loss and the train/test amb_HL
    for the ensemble formed of 100 trees
    
    bagger-the bagging object
    Xtr-training data
    ttr-training targets
    Xte-test data
    tte-test targets
    etest-array of test HL used for different forests
    etrain-array of train HL sed for different forests
    atest-array of test amb_HL
    atrain-array of train amb_HL
    '''
    
    pred_proba_train=bagger.predict_proba(Xtr)[:, 1].ravel()

    pred_proba_tst=bagger.predict_proba(Xte)[:, 1].ravel()
    
    #we can convert the prediction probability of belonging to the positive class,p,
    #to a score in the following way 4(p-1/2)
    clf_scores_tr=[4*(pred_proba_train[i]-1/2) for i in range(len(pred_proba_train))]
    clf_scores_tst=[4*(pred_proba_tst[i]-1/2) for i in range(len(pred_proba_tst))]

    etrain_HL.append(hinge_loss(ttr,clf_scores_tr))
    etest_HL.append(hinge_loss(tte,clf_scores_tst))
    
    clf_scores_tr_arr=[]
    clf_scores_tst_arr=[]
    
    trees=bagger.estimators_
    
    nr_trees=len(trees)
    
    c=list(np.ones(nr_trees)*(1/nr_trees))
    
    for i in range(nr_trees):
        tree=trees[i]
        
        prob_score_tr=tree.predict_proba(Xtr)[:, 1].ravel()
        prob_score_tst=tree.predict_proba(Xte)[:, 1].ravel()
     
        clf_scores_tr_arr.append([4*(prob_score_tr[i]-1/2) for i in range(len(prob_score_tr))])
        clf_scores_tst_arr.append([4*(prob_score_tst[i]-1/2) for i in range(len(prob_score_tst))])

    div_tr=getHingeLossDivEns(list(ttr),clf_scores_tr_arr, c)
    div_tst=getHingeLossDivEns(list(tte),clf_scores_tst_arr, c)
    
    atrain_HL.append(div_tr)
    atest_HL.append(div_tst)

    
def getDivValuesHL5(trees_list,Xtr,ttr,Xte,tte,etest_HL_5,etrain_HL_5,atest_HL_5,atrain_HL_5):
      '''
    Calculates the train/test hinge loss and the train/test amb_HL for the ensemble formed
    of 5 trees
    
    trees_list-list of 5 trees
    Xtr-training data
    ttr-training targets
    Xte-test data
    tte-test targets
    etest-array of test HL used for different forests
    etrain-array of train HL sed for different forests
    atest-array of test amb_HL
    atrain-array of train amb_HL
    '''
    
    clf_scores_tr_arr=[]
    clf_scores_tst_arr=[]
    
    nr_trees=len(trees_list)
    
    c=list(np.ones(nr_trees)*(1/nr_trees))
    
    for i in range(nr_trees):
        tree=trees_list[i]
        
        prob_score_tr=tree.predict_proba(Xtr)[:, 1].ravel()
        prob_score_tst=tree.predict_proba(Xte)[:, 1].ravel()

        clf_scores_tr_arr.append([4*(prob_score_tr[i]-1/2) for i in range(len(prob_score_tr))])
        clf_scores_tst_arr.append([4*(prob_score_tst[i]-1/2) for i in range(len(prob_score_tst))])


    div_tr=getHingeLossDivEns(list(ttr),clf_scores_tr_arr, c)
    div_tst=getHingeLossDivEns(list(tte),clf_scores_tst_arr, c)
    
    atrain_HL_5.append(div_tr)
    atest_HL_5.append(div_tst)
    
    etrain_HL_5.append(getHingeLossEns(ttr,clf_scores_tr_arr,c))
    etest_HL_5.append(getHingeLossEns(tte,clf_scores_tst_arr,c))
    
    
 
    
def getDivValuesChen(bagger,Xtr,ttr,Xte,tte,etest_Chen,etrain_Chen,atest_Chen,atrain_Chen):
     '''
    Calculates the train/test 0-1 loss and the train/test amb_01 for the ensemble formed
    of 100 trees
    
    bagger-the bagging object
    Xtr-training data
    ttr-training targets
    Xte-test data
    tte-test targets
    etest-array of test 0-1 loss used for different forests
    etrain-array of train 0-1 loss used for different forests
    atest-array of test amb_01
    atrain-array of train amb_01
    '''
          
    score_tree_tr=[]
    score_tree_tst=[]

    
    nr_trees=len(bagger.estimators_)
    
    estimators=bagger.estimators_
    
    for i in range(nr_trees):
        tree=estimators[i]
        
        score_tr=tree.predict(Xtr)
        score_tst=tree.predict(Xte)
        
        score_tr=[score_tr[i] if score_tr[i]==1 else -1 for i in range(len(score_tr)) ]
        score_tst=[score_tst[i] if score_tst[i]==1 else -1 for i in range(len(score_tst)) ]

        score_tree_tr.append(score_tr)
        score_tree_tst.append(score_tst)
    
    score_tree_tr=np.asarray(score_tree_tr)
    score_tree_tst=np.asarray(score_tree_tst)
  
    amtr,Htr=ambiguity(score_tree_tr,ttr)
    atrain_Chen.append(amtr) 
    
    amte,Hte=ambiguity(score_tree_tst,tte)
    atest_Chen.append(amte)  
    
    etrain_Chen.append(getErrorsClfs([Htr],ttr)[0])
    etest_Chen.append(getErrorsClfs([Hte],tte)[0])

    
def getDivValuesChen5(tree_list,Xtr,ttr,Xte,tte,etest_Chen_5,etrain_Chen_5,atest_Chen_5,atrain_Chen_5):
    
    '''
    Calculates the train/test 0-1 loss and the train/test amb_01 for the ensemble formed
    of 5 trees
    
    trees_list-list of 5 trees
    Xtr-training data
    ttr-training targets
    Xte-test data
    tte-test targets
    etest-array of test 0-1 loss used for different forests
    etrain-array of train 0-1 loss sed for different forests
    atest-array of test amb_01
    atrain-array of train amb_01
    '''
    
    score_tree_tr=[]
    score_tree_tst=[]
    
    nr_trees=len(tree_list)
    
    for i in range(nr_trees):
     
        tree=tree_list[i]
        
        score_tr=tree.predict(Xtr)
        score_tst=tree.predict(Xte)
        
        score_tr=[score_tr[i] if score_tr[i]==1 else -1 for i in range(len(score_tr)) ]
        score_tst=[score_tst[i] if score_tst[i]==1 else -1 for i in range(len(score_tst)) ]

        score_tree_tr.append(score_tr)
        score_tree_tst.append(score_tst)
    
    score_tree_tr=np.asarray(score_tree_tr)
    score_tree_tst=np.asarray(score_tree_tst)
  
    amtr,Htr=ambiguity(score_tree_tr,ttr)
    atrain_Chen_5.append(amtr) 
    
    amte,Hte=ambiguity(score_tree_tst,tte)
    atest_Chen_5.append(amte)  
    
    etrain_Chen_5.append(getErrorsClfs([Htr],ttr)[0])
    etest_Chen_5.append(getErrorsClfs([Hte],tte)[0])
    
def getEnsPredProba(h_proba):
    '''
    Returns the ensemble prediction, as a list of probabilities of belonging
    to the positive class
    
    h-proba-matrix of predictions of all classifiers
    '''
    
    nr_est=len(h_proba)
    H_proba=[sum(h_proba[:,j]) for j in range(len(h_proba[0]))]
    H_proba=[H_proba[i]/nr_est for i in range(len(H_proba))]
    
    return H_proba
    

def getMatrPredCE(estimators, X, t):
    
    '''
    Returns the matrix of predictions for all the classifiers
    
    estimators-list of all classifiers
    X-data
    t-target
    '''

   # y = np.sign(t)
 
    h_proba=np.asarray([estim.predict_proba(X)[:, 1].ravel() for estim in estimators])

    H_proba=getEnsPredProba(h_proba)
    
    return h_proba,H_proba

def div_CE(y,y_hat,c):
    
    '''
    Calculates the amb_CE of the ensemble for one pattern,
    see equation (10) from the paper
    
    y-class of one pattern
    y_hat-array of predictions of the classifiers for that pattern 
    c-array of weights
    
    '''
   
    eps=sys.float_info.epsilon
    div=0
    M=len(y_hat)
    
    if y!=0:
    
        S=np.sum(y_hat*c)
        if S==0:
            S=eps
        
        P=1
        for i in range(M):
            if y_hat[i]!=0:
                P*=y_hat[i]**c[i]
            else:
               #if the one of the probabilities is 0, then the product P, will be 0
               # and since in the formula of amb_CE=y*log(S/P), we will get a division
               #by 0 exception
               
               P*=eps**c[i] 
        
        #for the cases when the computer approximates wrongly the  product P
        # to be for ex 0.5000001 and the sum S=0.50, which we know can't be true
        #since the arithmetic mean is always >= to the geometric mean
        #https://en.wikipedia.org/wiki/Inequality_of_arithmetic_and_geometric_means
        
        if S<P:
            div+=0
        else:
        
            div+=y*math.log(S/P)
    
    if  y!=1:
        
        S=np.sum((1-y_hat)*c)
        if S==0:
            S=eps
        
        P=1
        for i in range(M):
            
            if y_hat[i]!=1:
                P*=(1-y_hat[i])**c[i]
            else:
               #if the one of the probabilities is 1, then the product P, will be 0
               # and since in the formula of amb_CE=y*log(S/P), we will get a division
               #by 0 exception
               P*=eps**c[i]  
        # when the computer wrongly approximates the product, same case as above
        if S<P:
            div+=0
        else:
        
            div+=(1-y)*math.log(S/P)
   
    
    return div


def div_ens(y,y_hat,c):
    
     '''
    Calculates the amb_CE of the ensemble for all patterns
    
    y-array of patterns
    y_hat-array of predictions of the classifiers for that pattern 
    c-array of weights
    '''
    
    S=0
    
    for i in range(len(y)):
        S+=div_CE(y[i],y_hat[:,i],c)
    
    N=len(y)
    
    S=S*1/N
    
    return S
	
def getHingeLossDiv(index_patt,targets,clf_scores,weights):
    
    '''
    Calculates the amb_HL of the ensemble for one pattern,given
    by index_patt
    see equation (12) from paper
    '''
    
    avg_err=0
    
    M=len(clf_scores)
    
    sum_1=0
  
    for i in range(M):
        sum_1+=weights[i]*(1-targets[index_patt]*clf_scores[i][index_patt])

    for i in range(M):
            
            avg_err+=weights[i]*max(0,1-targets[index_patt]*clf_scores[i][index_patt])

    hl_div=avg_err-max(0,sum_1)
  
    return hl_div

def getHingeLossDivEns(targets,clf_scores, weights):
    
    '''
    Calculates the amb_HL of the ensemble for all patterns,
    '''
    
    N=len(targets)
    
    S=0
  
    for i in range(N):
        
        hl_div=getHingeLossDiv(i,targets,clf_scores,weights)
        S+=hl_div
      
    
    return (S*1)/N

#read different datasets
def readDatabase(text_file,delimitator):
    ''' reads datasets like Australian, heart, sonar and Ionosphere'''
    f=open(text_file,'r')
    line=f.readline()
    X=[]
    y=[]

    while len(line)>0 :
        line=line.strip()
        if text_file.lower()in ['australian.txt','heart.txt']:
              delimitator=' '
       
        if text_file=='gmm5test.txt':
            line=line.replace('  ',',')#it usually has 3 spaces delimitator, here we put two spaces to replace bcz of the case when we have a minus
            delimitator=','
        z=line.split(delimitator)
        x=[float(y) for y in z[:-1]]
        x=np.asarray(x)
        X.append(x)
          
        if text_file=='Ionosphere_2.txt':
            if z[-1]=='g':
                y.append(1)
            else:
                y.append(0)
        elif text_file=='sonar.txt':
            if z[-1]=='R':
                  y.append(1)
            else:
                  y.append(0)
        elif text_file=='heart.txt':
            if int (z[-1])==1:
                y.append(0)
            else:
                y.append(1)
        elif text_file.lower()=='australian.txt':
            if int(z[-1])==0:
                y.append(0)
            else:
                y.append(1)
        else:
            y.append(float(z[-1]))
        line = f.readline()
    X=np.asarray(X)
    y=np.asarray(y)
    
    return X,y


def readWisconsinCancerDataset(text_file,delimitator):
    f=open(text_file,'r')
    line=f.readline()
    X=[]
    y=[]
     
    while len(line)>0 :
        line=line.strip()
       
        z=line.split(delimitator)
        x=[float(y) for y in z[2:]]
        x=np.asarray(x)
        X.append(x)
          
        if z[1]=='M':
            y.append(1)
        else:#B
            y.append(0)
              
        line=f.readline()
    X=np.asarray(X)
    y=np.asarray(y)
    
    return X,y
 
def readLiverDataset(text_file,delimitator):
    f=open(text_file,'r')
    line=f.readline()
    X=[]
    y=[]
     
    while len(line)>0 :
        line=line.strip()
       
        z=line.split(delimitator)
        x=[float(y) for y in z[:5]]
        x=np.asarray(x)
        X.append(x)
          
        if float(z[5])<3.0:
            y.append(0)
        else:#B
              y.append(1)
        
        line=f.readline()
    X=np.asarray(X)
    y=np.asarray(y)
    
    return X,y


def getXandYPerDataset(db,delimitator):
    if db=='cancer.txt':
        X,y=readWisconsinCancerDataset(db,delimitator)
    elif db=='liver.txt':
        X,y=readLiverDataset(db,delimitator)
    else:
        X,y=readDatabase(db,delimitator)
    return X,y

def readGerman(database_name):
    'read the whole data as train'
    X=[]
    y=[]
    
    try:
        f=open(database_name,'r')
        line=f.readline()
          
        while len(line)>0 :
                line=line.strip()
                z=line.split(',')
                x=[float(y) for y in z[:-1]]
         
                if int(z[-1])==1:
                    y.append(0)
                else:
                    y.append(1)
                    
                X.append(x)
                line=f.readline()
        return X,y
    except IOError:
          print('The database file specified does not exist')
    return []

def read_gender(filename):
    
    X=[]
    y=[]

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count=0
        for row in csv_reader:
            row_count+=1 #the first line contains the names of the columns
            if row_count>1:
                if row[-1]=='Male':
                    row[-1]=0
                else:
                    row[-1]=1
                row=[float(r) for r in row]
                X.append(row[:-1])
                y.append(row[-1])
    
    return X,y

def read_Star(filename):
    X=[]
 
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count=0
        for row in csv_reader:
            row_count+=1 #the first line contains the names of the columns
            if row_count>1:
                if '/' in row[4]:
                    index_str=row[4].index('/')
                    row[4]=row[4][:index_str]
                
                X.append(row)
    
    X=np.asarray(X)
    label_encoder = LabelEncoder()
    X[:,4] = label_encoder.fit_transform(X[:,4])
    X[:,5] = label_encoder.fit_transform(X[:,5])
    
    X_proces=[]
    y=[]
    
    for row in X:
        row=[float(r) for r in row]
        X_proces.append(row[:-1])
        y.append(row[-1])

    
    return X_proces,y   
    
    


