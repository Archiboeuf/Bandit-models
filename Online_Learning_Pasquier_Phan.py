# -*- coding: utf-8 -*-

import random as rd
import numpy as np
import pandas as pd
import scipy
import time
import matplotlib.pyplot as plt

K=5
T=2000


mus=np.random.random(K)
def borne(muchapeau_a,t,N_a,alpha=3,f=lambda t:np.log(t)):
    return(muchapeau_a+np.sqrt(alpha*f(t)/(2*N_a)))

def borne_hoeffding(muchapeau_a,t,N_a,delta=5/100):
    return(muchapeau_a+np.sqrt(np.log(t/delta)/(2*N_a)))
    
def smooth(mu,epsilon=10**-9): #résoud des problèmes comme des logarithmes de 0 ou division par 0 
    if mu>=1-epsilon:
        return(1-epsilon)
    elif mu<=epsilon:
        return(epsilon)
    else:
        return(mu)

def d(mu1,mu2): #divergence kl pour deux bernouillis de moyennes mu1 et mu2
    mu1,mu2=smooth(mu1),smooth(mu2)
    return(mu1*np.log(mu1/mu2)+(1-mu1)*np.log((1-mu1)/(1-mu2)))

#def borne_kl_UCB(muchapeau_a,t,N_a,f): #Section 2.2 page 7
#    if muchapeau_a>=1-10**-10: #simplification computationnelle
#        return(1)
#    return(scipy.optimize.fsolve(lambda q:N_a*d(muchapeau_a,q)-f(t),(muchapeau_a+1)/2)[0]) #Méthode de Newton avec initialisation à (muchapeau_a+1)/2
#on initialise à une valeur plus grande que muchapeau_a pour obtenir la racine plus grande que muchapeau_a    

def borne_kl_UCB(muchapeau_a,t,N_a,f): #Section 2.2 page 7
    if muchapeau_a>=1-10**-10: #simplification computationnelle
        return(1)
    def f_a_minimiser(q): #min de cette fonction correspond au u_a(t) page 7
        if N_a*d(muchapeau_a,q)<f(t):
            return(-q)
        else:
            return(np.inf)
    return(scipy.optimize.fminbound(f_a_minimiser,muchapeau_a,1))

def borne_kl_UCB_plus(muchapeau_a,t,N_a): #Section 2.2 page 7
    return(borne_kl_UCB(max(muchapeau_a,10**-10),t,N_a,lambda t: np.log(t/N_a))) #problème pour calculer divergence avec muchapeau_1=0

def g_a(x,mu1,mua): #page 11
    return(d(mu1,(mu1+x*mua)/(1+x))+x*d(mua,(mu1+x*mua)/(1+x)))
    
def inverse_g_a(y,mu1,mua):
    return(scipy.optimize.fsolve(lambda x:g_a(x,mu1,mua)-y,1)[0]) #Méthode de Newton
    
def F_mus(y,mus): #lemme 4 page 11
    res=0
    argmax=np.argmax(mus)
    mu_star=mus[argmax]
    for a in range(len(mus)):
        if a!=argmax:
            mua=mus[a]
            xay=inverse_g_a(y,mu_star,mua)
            mu2=(mu_star+xay*mua)/(1+xay)
            res+=d(mu_star,mu2)/d(mua,mu2)
    return(res)

def solve_dicho(f,xmin,xmax,target,epsilon=10**-4,max_iter=10**5): #résolution par dichotomie de f(x)=target pour f croissante
    if f(xmin)>target or f(xmax)<target:
        print('Erreur : pas de solution ou f non croissante')
        return((xmin+xmax)/2)
    else:
        t=0
        current_xmin,current_xmax=xmin,xmax
        x=(current_xmin+current_xmax)/2
        y=f(x)
        while abs(y-target)>epsilon and t<max_iter:
            t+=1
            if y>target:
                current_xmax=x
            else:
                current_xmin=x
            x=(current_xmin+current_xmax)/2
            y=f(x)
        if abs(y-target)>epsilon:
            print("Pas de convergence atteinte")
        return(y)


        
def solve_F_dicho(mus):
    mu_1=max(mus)
    mu_2=max([mu for mu in mus if mu!=mu_1])
    return(solve_dicho(lambda y : F_mus(y,mus),0,d(mu_1,mu_2),1))
    
def solve_F_Newton(mus):
    return(scipy.optimize.fsolve(lambda y : F_mus(y,mus)-1,0))

def calcul_ws(mus,methode=solve_F_Newton):
    ws=np.zeros(len(mus))
    mu_star=max(mus)
    L=len([mu for mu in mus if mu==mu_star])
    if L>1: # si il y a plusieurs bras optimaux
        for a,mu in enumerate(mus):
            if mu==mu_star:
                ws[a]=1/L
        return(ws)
    else:
        ystar=methode(mus)
        xs=[inverse_g_a(ystar,mu_star,mu) for mu in mus]
        ws=xs/sum(xs)
        return(ws)
        
    

def UCB(mus,T,calcul_borne=borne_kl_UCB_plus):
    gain=0
    K=len(mus)
    Ns=[0]*K
    muschapeaux=np.array([0.0]*K)
    for t in range(K):
        Ns[t]+=1 #le bras t est utilisé
        Xt=rd.random()<mus[t] #gain de l'action
        gain+=Xt
        muschapeaux[t]=Xt
    bornes=[float(calcul_borne(muschapeaux[a],t,Ns[a])) for a in range(K)] #borne sup de l'intervalle de confiance
    columns=(['t','gain'] 
            + ['N_' + str(a) for a in range(K)] 
            + ['muchapeau_' + str(a) for a in range(K)]
            + ["borne_" + str(a) for a in range(K)])
    df=pd.DataFrame(index=[t for t in range(K-1,T)],columns=columns) #pour plot l'historique
    df.loc[K-1]=pd.Series(np.concatenate([[t],[gain],Ns,muschapeaux,bornes]),index=columns)
    for t in range(K,T):
        a=np.argmax(bornes) #on choisi le bras dont la borne sup est la plus élevée
        Xt=rd.random()<mus[a] #gain de l'action
        gain+=Xt
        muschapeaux[a]=(muschapeaux[a]*Ns[a]+Xt)/(Ns[a]+1) #mise à jour de l'estimation du gain moyen
        Ns[a]+=1
        bornes=[float(calcul_borne(muschapeaux[x],t,Ns[x])) for x in range(K)] #recalcul des bornes
        df.loc[t]=pd.Series(np.concatenate([[t],[gain],Ns,muschapeaux,bornes]),index=columns)
    return(df,gain,Ns,muschapeaux,bornes)

df,gain,Ns,muschapeaux,bornes=UCB(mus,T,borne)
#df,gain,Ns,muschapeaux,bornes=UCB(mus,T,borne_kl_UCB_plus)

df[["borne_" + str(a) for a in range(K)]].plot()
plt.title("Evolution des bornes")
plt.xlabel('t')
plt.ylabel('borne')
#plt.savefig('C:\\Users\\benji\\Desktop\\Cours\\Apprentissage en ligne et agregation\\evolution bornes.png')

df[["muchapeau_" + str(a) for a in range(K)]].plot()
plt.title("Evolution des moyennes empiriques")
plt.xlabel('t')
plt.ylabel('moyenne empirique')
#plt.savefig('C:\\Users\\benji\\Desktop\\Cours\\Apprentissage en ligne et agregation\\evolution moyennes.png')

gain/T/max(mus) #doit converger vers 1 quand T tend vers l'infini
df['regret']=df['t']*max(mus)-df['gain']

df['regret'].plot()
plt.title("Evolution du regret")
plt.xlabel('t')
plt.ylabel('Regret')
#plt.savefig('C:\\Users\\benji\\Desktop\\Cours\\Apprentissage en ligne et agregation\\Regret.png')

(df['regret']/df['t']).plot()

def track(mus,T,methode=solve_F_Newton):
    gain=0
    K=len(mus)
    Ns=[0]*K
    muschapeaux=np.array([0.0]*K)
    for t in range(K):
        Ns[t]+=1
        Xt=rd.random()<mus[t]
        gain+=Xt
        muschapeaux[t]=Xt
    ws=calcul_ws(muschapeaux,methode)
    columns=(['t','gain'] 
            + ['N_' + str(a) for a in range(K)] 
            + ['muchapeau_' + str(a) for a in range(K)]
            + ["w_" + str(a) for a in range(K)])
    df=pd.DataFrame(index=[t for t in range(K-1,T)],columns=columns)
    df.loc[K-1]=pd.Series(np.concatenate([[t],[gain],Ns,muschapeaux,ws]),index=columns)
    for t in range(K,T):
        Ft=[a for a in range(K) if Ns[a]<np.sqrt(t)-K/2] #tracking rule page 12
        if len(Ft)>0:
            a=Ft[0] #on force l'exploration d'un bras peu exploré
        else:
            a=np.argmax(ws-np.array(Ns)/t) #on prend le bras dont la proportion d'exploration est la plus en dessous de la proportion optimale
        Xt=rd.random()<mus[a]
        gain+=Xt
        muschapeaux[a]=(muschapeaux[a]*Ns[a]+Xt)/(Ns[a]+1)
        Ns[a]+=1
        ws=calcul_ws(muschapeaux,methode)
        df.loc[t]=pd.Series(np.concatenate([[t],[gain],Ns,muschapeaux,ws]),index=columns)
    return(df,gain,Ns,muschapeaux,ws)
    
    
#df,gain,Ns,muschapeaux,ws=track(mus,300,solve_F_dicho)
df,gain,Ns,muschapeaux,ws=track(mus,300,solve_F_Newton)

df[["w_" + str(a) for a in range(K)]].plot()
df[["muchapeau_" + str(a) for a in range(K)]].plot()
gain/T/max(mus)
df['regret']=df['t']*max(mus)-df['gain']
df['regret'].plot()
(df['regret']/df['t']).plot()


#Track and stop
def Z_hat(mu_a,Na,mu_b,Nb): #page 13 après l'expression (8)
    mu2=(Na*mu_a+Nb*mu_b)/(Na+Nb)
    return(Na*d(mu_a,mu2)+Nb*d(mu_b,mu2))

def test_chernoff(t,mus,Ns,K,delta=5/100): #lemme 6 page 14
    seuil=np.log(2*(K-1)*t/delta)
    for a in range(K-1):
        for b in range(a+1,K):
            if a!=b:
                mu_a,Na,mu_b,Nb=mus[a],Ns[a],mus[b],Ns[b]
                if Z_hat(mu_a,Na,mu_b,Nb)>seuil:
                    return(True)
    return(False)
                
def track_and_stop(mus,T,methode=solve_F_Newton,delta=5/100,max_iter=20000):
    gain=0
    K=len(mus)
    Ns=[0]*K
    muschapeaux=np.array([0.0]*K)
    for t in range(K):
        Ns[t]+=1
        Xt=rd.random()<mus[t]
        gain+=Xt
        muschapeaux[t]=Xt
    ws=calcul_ws(muschapeaux,methode)
    columns=(['t','gain'] 
            + ['N_' + str(a) for a in range(K)] 
            + ['muchapeau_' + str(a) for a in range(K)]
            + ["w_" + str(a) for a in range(K)])
    df=pd.DataFrame(index=[K-1],columns=columns)
    df.loc[K-1]=pd.Series(np.concatenate([[t],[gain],Ns,muschapeaux,ws]),index=columns)
    #identique à track jusqu'ici
    stop=test_chernoff(t,muschapeaux,Ns,K,delta)
    while not (stop or t>=max_iter):
        t+=1
        Ft=[a for a in range(K) if Ns[a]<np.sqrt(t)-K/2] #tracking rule page 12
        if len(Ft)>0:
            a=Ft[0] #on force l'exploration d'un bras peu exploré
        else:
            a=np.argmax(ws-np.array(Ns)/t) #on prend le bras dont la proportion d'exploration est la plus en dessous de la proportion optimale
        Xt=rd.random()<mus[a]
        gain+=Xt
        muschapeaux[a]=(muschapeaux[a]*Ns[a]+Xt)/(Ns[a]+1)
        Ns[a]+=1
        ws=calcul_ws(muschapeaux,methode) #mise à jour des poids
        df.loc[t]=pd.Series(np.concatenate([[t],[gain],Ns,muschapeaux,ws]),index=columns)
        stop=test_chernoff(t,muschapeaux,Ns,K,delta)
    if t>=max_iter:
        print("maximum d'itérations atteint")
    return(df,gain,Ns,muschapeaux,ws)


#df,gain,Ns,muschapeaux,ws=track_and_stop(mus,T,solve_F_dicho,max_iter=2000)
df,gain,Ns,muschapeaux,ws=track_and_stop(mus,T,solve_F_Newton,max_iter=2000)

df[["w_" + str(a) for a in range(K)]].plot()
df[["muchapeau_" + str(a) for a in range(K)]].plot()
gain/df.shape[0]/max(mus)
df['regret']=df['t']*max(mus)-df['gain']
df['regret'].plot()
(df['regret']/df['t']).plot()


#########   Evaluation des résultats (long à exécuter)   #########

#les UCB
K=5
N=30
T=2000
cols=(['R_UCB','R_kl_UCB+','temps_UCB','temps_kl_UCB+','bon_bras_UCB','bon_bras_kl_UCB+']
    + ['mu_'+str(a) for a in range(K)] 
    + ['muchapeau_'+str(a)+'_'+method for method in ['UCB','kl_UCB'] for a in range(K)])
comparaison_UCB=pd.DataFrame(columns=cols)
for t in range(N):
    print(t)
    mus=np.random.random(K) #on simule de nouvelles moyennes à chaque fois
    #UCB
    start=time.time()
    df,gain,Ns,muschapeaux_UCB,bornes=UCB(mus,T,borne)
    temps_UCB=time.time()-start #mesure du temps
    bon_bras_UCB=np.argmax(mus)==np.argmax(muschapeaux_UCB)
    R_UCB=T*max(mus)-gain
    #kl-UCB+
    start=time.time()
    df,gain,Ns,muschapeaux_kl_UCB,bornes=UCB(mus,T,borne_kl_UCB_plus)
    temps_kl_UCB=time.time()-start
    bon_bras_kl_UCB=np.argmax(mus)==np.argmax(muschapeaux_kl_UCB)    
    R_kl_UCB=T*max(mus)-gain
     #on réordonne les mus pour commodité de lecture ; on ne le fait pas avant à cause du fait que
     #l'algorithme choisit par défaut le premier bras en cas d'ex-aequo
    ordre={}
    for i in range(K):
        ordre[mus[i]]=i
    mus.sort()
    muschapeaux_UCB=[muschapeaux_UCB[ordre[mu]] for mu in mus]
    muschapeaux_kl_UCB=[muschapeaux_kl_UCB[ordre[mu]] for mu in mus]
    #stockage
    comparaison_UCB.loc[t]=pd.Series(np.concatenate(([R_UCB,R_kl_UCB,temps_UCB,temps_kl_UCB,
           bon_bras_UCB,bon_bras_kl_UCB],mus,muschapeaux_UCB,muschapeaux_kl_UCB)),index=cols)

#comparaison_UCB.to_csv('C:\\Users\\benji\\Desktop\\Cours\\Apprentissage en ligne et agregation\\comparaison UCB.csv',index=False)    
comparaison_UCB['R_UCB'].mean() #76
comparaison_UCB['R_UCB'].std() #18
comparaison_UCB['R_kl_UCB+'].mean() #42
comparaison_UCB['R_kl_UCB+'].std() #126
#kl-UCB+ meilleur regret en moyenne mais plus grosse variabilité
comparaison_UCB['temps_UCB'].mean() #4
comparaison_UCB['temps_kl_UCB+'].mean() #12
#kl-UCB+ presque 3 fois plus long à exécuter

np.sqrt(K*T)/20 #5 minore inf_pi sup_mu Risque
comparaison_UCB['R_UCB'].max() #104
comparaison_UCB['R_kl_UCB+'].max() #514

comparaison_UCB['bon_bras_UCB'].mean() #0.93
comparaison_UCB['bon_bras_kl_UCB+'].mean() #0.87

#Conclusion UCB : le kl-UCB+ a un regret plus faible en moyenne mais il a une plus grande variabilité
# Il explore moins souvent les bras avec une faible moyenne empirique ce qui lui donne une plus
# grande probabilité de se tromper. Il est également plus long à exécuter (il faut optimiser une
# fonction). Par ailleurs, dans le cas de lois de bernouillis, l'expression de la divergence kl
# est simple mais il se peut que ce ne soit pas toujours le cas.


#Track and stop
K=5
N=30
T=2000
cols=(['R_dicho','R_Newton','temps_dicho','temps_Newton','iteration_dicho','iteration_Newton',
       'bon_bras_dicho','bon_bras_Newton']
    + ['mu_'+str(a) for a in range(K)] 
    + ['muchapeau_'+str(a)+'_'+method for method in ['dicho','Newton'] for a in range(K)])


comparaison_tas=pd.DataFrame(columns=cols)
for t in range(N):
    print(t)
    mus=np.random.random(K) #on simule de nouvelles moyennes à chaque fois
    #dichotomie
    start=time.time()
    df,gain,Ns,muschapeaux_dicho,ws=track_and_stop(mus,T,solve_F_dicho,max_iter=500)
    temps_dicho=time.time()-start
    iteration_dicho=df.shape[0]
    R_dicho=iteration_dicho*max(mus)-gain
    bon_bras_dicho=np.argmax(mus)==np.argmax(muschapeaux_dicho)
    #Newton
    start=time.time()
    df,gain,Ns,muschapeaux_Newton,ws=track_and_stop(mus,T,solve_F_Newton,max_iter=500)
    temps_Newton=time.time()-start
    iteration_Newton=df.shape[0]
    R_Newton=iteration_Newton*max(mus)-gain
    bon_bras_Newton=np.argmax(mus)==np.argmax(muschapeaux_Newton)
    ordre={} #on réordonne les mus
    for i in range(K):
        ordre[mus[i]]=i
    mus.sort()
    muschapeaux_dicho=[muschapeaux_dicho[ordre[mu]] for mu in mus]
    muschapeaux_Newton=[muschapeaux_Newton[ordre[mu]] for mu in mus]
    comparaison_tas.loc[t]=pd.Series(np.concatenate((
            [R_dicho,R_Newton,temps_dicho,temps_Newton,iteration_dicho,iteration_Newton,
             bon_bras_dicho,bon_bras_Newton],mus,muschapeaux_dicho,muschapeaux_Newton)),index=cols)
    
#comparaison_tas.to_csv('C:\\Users\\benji\\Desktop\\Cours\\Apprentissage en ligne et agregation\\comparaison tas.csv',index=False)    

#Il semble y avoir des problemes dans la dichotomie de temps en temps


comparaison_tas['iteration_dicho'].mean() #208
comparaison_tas['iteration_Newton'].mean() #164
comparaison_tas['iteration_dicho'].std() #162
comparaison_tas['iteration_Newton'].std() #160
comparaison_tas['temps_dicho'].mean() #11
comparaison_tas['temps_Newton'].mean() #11
#Plus rapide avec méthode de Newton

comparaison_tas['bon_bras_dicho'].mean() #0.7
comparaison_tas['bon_bras_Newton'].mean() #0.8
#identification plus précise avec Newton

(comparaison_tas['R_dicho']/comparaison_tas['iteration_dicho']).mean() #0.31
(comparaison_tas['R_dicho']/comparaison_tas['iteration_dicho']).std() #0.15
(comparaison_tas['R_Newton']/comparaison_tas['iteration_Newton']).mean() #0.11
(comparaison_tas['R_Newton']/comparaison_tas['iteration_Newton']).std() #0.07
#regret moins élevé avec Newton

#Conclusion track and stop : méthode de Newton semble meilleure sur tous les aspects que la dichotomie (qui signale
#de temps en temps des erreurs). Le bon bras est identifié moins souvent que prévu.


#Conclusion générale : on oberve bien que les méthodes UCB ont un regret faible alors que le track
# and stop ne permet pas d'avoir un regret aussi faible, comme le prédit la théorie.
# Le track and stop permet donne une règle d'arrêt parcimonieuse ce qui permet de faire
# peu d'itérations avant de pouvoir essayer de deviner quel bras est le bon avec une bonne
# probabilité d'avoir raison. Cependant, chaque itération de l'algorithme track and stop est
# plus longue à exécuter que dans un algorithme UCB (plusieurs programme d'optimisation à résoudre,
# inversion de fonctions ..). Résultat : dans un même lapse de temps, un algo UCB peut faire plus
# d'itérations ce qui lui permet de deviner le bon bras avec autant de probabilité, voire plus, 
# qu'un algorithme track and stop. (Il faut cependant prendre en compte d'autres aspects autre que
# computationels ; dans la détection d'un meilleur traitement où il faut attendre longtemps pour
# observer le gain, la vitesse d'exécution d'algorithme importe peu.)


#Notes : parfois il y a des problèmes computationnels (non convergence des programmes 
#d'optimisation, division par 0 ...)
