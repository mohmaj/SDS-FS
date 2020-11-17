"""
The code is originally written by Haya Alhakbani for 
H. Alhakbani and M. M. al-Rifaie, "Feature selection using stochastic diffusion search," in Proceedings of the Genetic and Evolutionary Computation Conference. ACM, 2017, pp. 385-392.

Updated by: Mohammad Majid al-Rifaie, November 2020
"""

import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import metrics
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

testRun = True

score_store= {}
def SDS(features_size,cost_function, data, maxIteration=50, population_size = 100):
    VarSize = data.shape[1]-1
    search_space_size = math.factorial(VarSize)/(math.factorial(features_size)*math.factorial(VarSize-features_size))
    print("Search space size: "+str(search_space_size))

	#agent class
    class Agent(object):
        def __init__(self):
            self.set_random_hypotesis()
            self.active = False

        def set_random_hypotesis(self):
            self.hypotesis = [1]*features_size + [0]*(VarSize-features_size)
            np.random.shuffle(self.hypotesis)

        def set_hypotesis_with_offset(self, new_hypotesis):
            self.hypotesis[:] = new_hypotesis #copy good hypotesis
            zero_set = False
            one_set = False
            stored = -1
            z=VarSize-1
            while not zero_set or not one_set:
                r = random.randint(0,z)
                if(self.hypotesis[r]==1 and stored!=r and not zero_set):
                    self.hypotesis[r]=0
                    zero_set = True
                    stored = r
                r2 = random.randint(0,z)
                if(self.hypotesis[r2]==0 and stored!=r2 and not one_set):
                    self.hypotesis[r2]=1
                    one_set = True
                    stored = r2

        def calculate(self,with_metric=True):
            if self.active==True:
                self.score

            indexes = []
            for j in range(0,VarSize):
                if(self.hypotesis[j]==1):
                    indexes.append(j)
            indexes.append(VarSize)
            subdata = data[:,indexes]
            #print('Indices selected:' , indexes[:len(indexes)-1])
            
            if(with_metric):
                self.score = cost_function(subdata,True)
            else:
                self.score = cost_function(subdata,True)

        def cal(self,  hypo):
            indexes = []
            for j in range(0,VarSize-1):
                if(hypo[j]==1):
                    indexes.append(j)
            indexes.append(VarSize)
            subdata = data[:,indexes]

            score = cost_function(subdata)
            return score


    # INITIALIZATION
    def initial():
        global population
        population=[]
        for ag in range(population_size):
            population.append(Agent())
        for ag in population:
            ag.calculate()

    #TEST
    def test():
        for ag in population:
            s = np.random.choice(population)
            while(s == ag):
                s = np.random.choice(population)
            another_agent = s
            if(ag.score >= another_agent.score):
                ag.active = True
            else:
                ag.active = False

    #DIFFUSION
    def diffusion():
        for ag in population:
             if (ag.active==False):
                #print('ag is inactive and the score is:', ag.score,'and its hypo is:', ag.hypotesis)
                another_agent= np.random.choice(population)
                while(another_agent==ag):
                    another_agent=np.random.choice(population)
                if(another_agent.active):
                    #print('The other agent hypo is', another_agent.hypotesis)
                    ag.set_hypotesis_with_offset(another_agent.hypotesis)
                    ag.calculate()
                    #print('After the offset:', ag.score, ', with hypo:', ag.hypotesis)
                else:
                    ag.set_random_hypotesis()
                    ag.calculate()
                    #print('The score of random hypo:', ag.score, ' and the new hypo is:', ag.hypotesis)

             else:
                 #print('Ag is active and its score is', ag.score)
                 another_agent= np.random.choice(population)
                 while(another_agent==ag):
                    another_agent=np.random.choice(population)
                 if((another_agent.active) and (another_agent.hypotesis == ag.hypotesis)):
                    ag.active=False
                    ag.set_random_hypotesis()
                    ag.calculate()
                    #print('The score after context sensitive diffusion is: ', ag.score)

    # #MAIN
    trial=10
    if testRun: trial = 2
    maxAcc=-1
    trial_best=[]
    for trialCounter in range(trial):
        initial()
        maxAccForit=-1
        best_it=[]
        print('\n\n====> Trial:', trialCounter)
        for iteration in range(maxIteration):
            print('\n==>Iteration:', iteration)
            it=-1
            for ag in population:
                if ag.score>it:
                    it=ag.score

            for ag in population:
                #print(ag.score)
                #print(ag.active)
                if (ag.score > maxAccForit):
                    maxAccForit=ag.score
                if(ag.score>maxAcc):
                    maxAcc=ag.score
                    best=ag.hypotesis

            test()
            diffusion()

            print( 'Best accuracy in iteration: \t', maxAccForit)
            best_it.append(maxAccForit)#for ploting the max acc at each iteration
            print('Best overall accuracy:\t\t', maxAcc)

        trial_best.append(max(best_it)) #here the trial ends


    print('\nBest accuracy in each trial:', trial_best)

    print('Time elapsed:', (time.time() - start_time))
    t=[]
    for i in range(trial):
        t.append(i)
    tr=np.array(t)
    tra=np.array(trial_best)
    plt.plot(tr,tra)
    plt.scatter(tr,tra, s=50)
    plt.xlabel('Trials')
    plt.xticks(range(0,len(t)))
    plt.ylabel('Accuracy')
    plt.title('Accuracy in each Trial')
    plt.show()
    print('\n======== FINAL RESULTS ======== ')
    return best

#to calculate the run time
start_time = time.time()


#Fintess function
def svm_accuracy(data, additional_metrics = False):
    np.random.shuffle(data)
    np.random.shuffle(data)
    train, test = train_test_split(data, test_size = 0.2)
    clf = svm.SVC(kernel='rbf', gamma='scale')
    clf.fit(train[:,0:-1], train[:,-1])
    accurancy = clf.score(test[:,0:-1],test[:,-1])
    if(additional_metrics):
        predicted = clf.predict(test[:,0:-1])
        CM = metrics.confusion_matrix(test[:,-1], predicted)
        #print('the Confusion Matrix')
        #print(CM)
        #TN = CM[0][0]
        #FN = CM[1][0]
        #TP = CM[1][1]
        #FP = CM[0][1]
        #print("Additional metrics: ")
        #print("TP "+str(TP))
        #print("FP "+str(FP))
        #print("TN "+str(FN))
        #print("FN "+str(FN))
        #sensitivity = TP / (TP + FN)
        #print(sensitivity)
        #specificity = TN / (FP + TN)
        #print("Specificity "+str(specificity))
        #print('Accuracy', accurancy)
    return accurancy


print("SVM - SDS features selection")
my_data = genfromtxt('bupa.csv', delimiter=',')
x_norm = my_data[:,0:-1] / my_data[:,0:-1].max(axis=0)
my_data =np.concatenate((x_norm,np.array([my_data[:,-1]]).T), axis=1)
np.random.shuffle(my_data)
print('Originial data shape:', my_data.shape)

# For all features
print("Accuracy for all features: "+str(abs(svm_accuracy(my_data, True))) + "\n")

#For selected features
features_size = int((my_data.shape[1]-1)/2) # half the original features
maxIteration = 50 
if testRun: maxIteration = 30
population_size = 100
if testRun: population_size = 30
print("Selected features "+str(SDS(features_size,svm_accuracy,my_data,maxIteration,population_size)))
