# -*- coding: utf-8 -*-
"""
https://arxiv.org/abs/1709.09002
"""
import random
import math
import numpy as np
import networkx as nx
import scipy
from math import exp

def create_A_matrix(data,nbr_team):
    A=np.zeros([nbr_team,nbr_team])
    nodes={}
    compteur=0
    for row in data:
        if row[0] not in nodes:
            nodes[row[0]]=compteur
            compteur+=1
        if row[1] not in nodes:
            nodes[row[1]]=compteur
            compteur+=1
        A[nodes[row[0]],nodes[row[1]]]+=int(row[2])
        
    return A,nodes

def create_A_matrix_with_nodes(data,nbr_team,nodes):
    A=np.zeros([nbr_team,nbr_team])
    for row in data:
        A[nodes[row[0]],nodes[row[1]]]+=int(row[2])
    return A

def build_from_dense(A,nbr_team, alpha, l0, l1):
    """
    Given as input a 2d numpy array, build the matrices A and B to feed to the linear system solver for SpringRank.
    """
    D_in = np.sum(A, 0)
    D_out = np.sum(A, 1)

    D1 = D_in + D_out           # to be seen as diagonal matrix, stored as 1d array
    D2 = l1 * (D_out - D_in)    # to be seen as diagonal matrix, stored as 1d array

    if alpha != 0.:
        B = np.ones(nbr_team) * (alpha * l0) + D2
        A = - (A + A.T)
        A[np.arange(nbr_team), np.arange(nbr_team)] = alpha + D1 + np.diagonal(A)
        
        

    else:
        last_row_plus_col = (A[nbr_team - 1, :] + A[:, nbr_team - 1]).reshape((1, nbr_team))
        A = (A + A.T)
        A += last_row_plus_col
        
        A=D1-A

        A[np.arange(nbr_team), np.arange(nbr_team)] = A.diagonal() + D1
        D3 = np.ones(nbr_team) * (l1 * (D_out[nbr_team - 1] - D_in[nbr_team - 1]))  # to be seen as diagonal matrix, stored as 1d array
        B = D2 + D3

    return A, B

def linear_solver(A, B):
    sol=scipy.linalg.solve(A,B)

    return sol

def ranking(A,nbr_team,alpha=0):
    l0=1.
    l1=1.
    A, B = build_from_dense(A,nbr_team, alpha, l0, l1)

    rank = linear_solver(A, B)

    return rank

def proba_win(score1,score2,beta=1,rounding=-1):
#    calculate probability that team with score1 win over team with score 2
    result=1/(1+exp(-beta*(score1-score2)))
    if rounding==-1:
        return result
    else:
        return round(1/(1+exp(-beta*(score1-score2))),rounding)

def spring_rank_hero(data_tour,nbr_hero,dico):
    A,nodes=create_A_matrix(data_tour,nbr_hero)
    rank=ranking(A,nbr_hero,alpha=0)
    rank-=min(rank)

    order=np.argsort(rank)
    sol_ordered=np.sort(rank)
    reverse_nodes={value : key for (key, value) in nodes.items()}
    
    name=[]
    for x in order:
        name.append(reverse_nodes[x])
    for i in range(len(name)):
        if name[i] in dico:
            name[i]=dico[name[i]]
            
    name.reverse()
    sol_ordered=np.flip(sol_ordered)
    
    name_score=[]
    for i in range(len(name)):
        name_score.append([name[i],sol_ordered[i]])
        
    beta=extract_beta(rank,nbr_hero,A)
    return name_score,beta

def extract_beta(rank,nbr_hero,A,start=0,end=10,step=100,power=2):
    A_test=np.ones([nbr_hero,nbr_hero])*-1
    for i in range(nbr_hero):
        for j in range(nbr_hero):
            if j>i:
                if A[i,j]==0 and A[j,i]==0:
                    pass
                else:
                    A_test[i,j]=A[i,j]/max(1,A[i,j]+A[j,i])
    keep_diff=[]
    for B in range(start,end*step):
        beta=B/step
        #print(beta,"/",end)
        score=0
        for i in range(nbr_hero):
            for j in range(nbr_hero):
                if j>i:
                    if A_test[i,j]!=-1:
                        score+=abs((A_test[i,j]-proba_win(rank[i],rank[j],beta)))**power
        keep_diff.append(score)
    B_min=keep_diff.index(min(keep_diff))
    beta=B_min/step
    return beta

def link_ranking_different_beta(rank_beta,rank,target_beta):
    # rank is either the solution from spring_rank ranking or the ordered score with name
    multiplier=rank_beta/target_beta
    if len(np.shape(rank))==1:
        return rank*multiplier
    rank=[[x[0],x[1]*multiplier] for x in rank]
    return rank

def see_gain(changement_val,beta):
    return 1/(1+exp(-beta*changement_val))-0.5

    
    
    
    
    
    
    
    
    
