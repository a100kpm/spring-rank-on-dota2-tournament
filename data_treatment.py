# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
from spring_rank import *
from get_data import *


def create_hero_dico(path='dota_hero.csv'):
#

# output the dictionary containing all heroes from the file in path
    hero=pd.read_csv(path)
    dico=dict()
    for x in hero.iterrows():
        val=x[1]
        dico[val.hero_id]=val.hero_name
    return dico

def hero_set(data_tour):
# data_tour -> the data from create_hero_data_for_springrank

# output the set of all heroes and the number of heroes
    count_hero=set()
    for x in data_tour:
        count_hero.add(x[0])
        count_hero.add(x[1])
        
    nbr_hero=len(count_hero)
    return count_hero,nbr_hero

def create_data(p1,p2,val=1):
# p1,p2 -> list of picks of each team, p1 is winner, p2 is looser
# val -> ponderation value (must be integer)

# output basic chunk of data usefull to create the matrix A of spring rank
    rez=[]
    for x in p1:
        for y in p2:
            rez.append([x,y,val])
    return rez

def create_hero_data_for_springrank(path='tournament.csv',winrate=5000):
# winrate -> 100*winrate from the tournament (we HAVE to keep integer value!)

# output the data of the tournament for spring rank with ponderation regarding the side if winrate var is changed
    data=pd.read_csv(path)
    data_tour=[]
    winrate_radiant=10000-winrate
    winrate_dire=winrate
    for x in data.iterrows():
        pick_radiant=list(map(int,x[1].pick_radiant[1:-1].split(',')))
        pick_dire=list(map(int,x[1].pick_dire[1:-1].split(',')))
        if x[1].win_radiant==True:
            val=create_data(pick_radiant,pick_dire,winrate_radiant)
        else:
            val=create_data(pick_dire,pick_radiant,winrate_dire)
        
        data_tour.extend(val)
    return data_tour

def treat_data(data_tour,remove_inf=5):
# data_tour -> the data from create_hero_data_for_springrank
# remove_inf -> number of games threshold of removal of heroes

# output the data of the tournament without heroes who were present less than remove_inf matches
    count_=dict()
    for x in data_tour:
        if x[0] in count_:
            count_[x[0]]+=1
        else:
            count_[x[0]]=1
        if x[1] in count_:
            count_[x[1]]+=1
        else:
            count_[x[1]]=1
    set_remove=set()
    for x in count_:
        if count_[x]<=remove_inf*5:
            set_remove.add(x)
    new_data_tour=[]
    for x in data_tour:
        if x[0] in set_remove or x[1] in set_remove:
            pass
        else:
            new_data_tour.append(x)
    return new_data_tour

def treat_data_post(name_score,data_tour,dico,remove_inf=5):
# name_score -> spring rank result obtained from create_hero_ranking
# data_tour -> the data from create_hero_data_for_springrank
# dico -> dictionary of all heroes
# remove_inf -> number of games threshold of removal of heroes

# output the spring ranking without heroes who were present less than remove_inf matches
    count_=dict()
    for x in data_tour:
        if x[0] in count_:
            count_[x[0]]+=1
        else:
            count_[x[0]]=1
        if x[1] in count_:
            count_[x[1]]+=1
        else:
            count_[x[1]]=1
    set_remove=set()
    for x in count_:
        if count_[x]<=remove_inf*5:
            set_remove.add(dico[x])
    name_score=[x for x in name_score if x[0] not in set_remove]
    return name_score

def ranking_team(path='tournament.csv'):
#

# the spring ranking of every teams
    data_tour_team=create_team_data_for_springrank(path=path)
    count_team,nbr_team=team_set(data_tour_team)
    
    
    A_team,nodes_team=create_A_matrix(data_tour_team, nbr_team)
    rank_team=ranking(A_team,nbr_team,alpha=0)
    rank_team-=min(rank_team)
    rank_team/=max(rank_team)
    
    order_team=np.argsort(rank_team)
    sol_ordered_team=np.sort(rank_team)
    reverse_nodes_team={value : key for (key, value) in nodes_team.items()}
    
    
    name_team=[]
    for x in order_team:
        name_team.append(reverse_nodes_team[x])
            
    name_team.reverse()
    sol_ordered_team=np.flip(sol_ordered_team)
    
    name_score_team=[]
    for i in range(len(name_team)):
        name_score_team.append([name_team[i],sol_ordered_team[i]])
    beta=extract_beta(rank_team,nbr_team,A_team)
    return name_score_team,beta

def create_team_data_for_springrank(path='tournament.csv'):
#

# output the basic data needed to generate the matrix A of spring rank in order to rank teams
    data=pd.read_csv(path)
    data_tour_team=[]
    for x in data.iterrows():
        radiant=x[1].radiant
        dire=x[1].dire
        win=x[1].win_radiant
        if win==True:
            val=[radiant,dire,1]
        else:
            val=[dire,radiant,1]
        data_tour_team.append(val)
    return data_tour_team

def team_set(data_tour_team):
# data_tour_team -> the data from create_team_data_for_springrank

# output the set of all teams and the number of teams
    count_team=set()
    for x in data_tour_team:
        count_team.add(x[0])
        count_team.add(x[1])
        
    nbr_team=len(count_team)
    return count_team,nbr_team

def generate_data_team_side_factor(path='tournament.csv'):
#

# output the spring factor associated with each matches and the winrate of the tournament
    data=pd.read_csv(path)
    name_score_team,beta_team=ranking_team(path)
    name_score_team=pd.DataFrame(name_score_team)
    winrate = get_winrate_from_tour(data)
    
    data=data.merge(name_score_team, how='left', left_on=['radiant'], right_on=[0])
    data.drop(columns=[0],inplace=True)
    data.rename(columns={1: 'radiant_score'},inplace=True)
    data=data.merge(name_score_team, how='left', left_on=['dire'], right_on=[0])
    data.drop(columns=[0],inplace=True)
    data.rename(columns={1: 'dire_score'},inplace=True)
    data['spring_rank_factor']= data.apply(lambda x: int(100*proba_win(x.radiant_score, x.dire_score,beta_team,2)), axis=1)
    return data,winrate

def create_hero_ranking(team_weight=True,side_weight=True,remove_inf=0,remove_post=True,team_list=[],path='tournament.csv'):
# team_weight -> take into account the level of the team
# side_weight -> take into account the winrate of each side
# remove_inf -> remove heroes with less than remove_inf+1 match
# remove_post -> remove heroes before or after treatment
# team_list -> remove match from team in the list
    
# output the spring ranking of every heroes
    data,side_winrate=generate_data_team_side_factor(path=path)
    data=data[~ (data.radiant.isin(team_list)|data.dire.isin(team_list)) ]
    data_tour=[]
    if side_weight==False:
        side_winrate=50
        
    for x in data.iterrows():
        if team_weight==True:
            multiplier_factor=int(100*get_multiplier_factor(side_winrate,x[1].spring_rank_factor))
        else:
            multiplier_factor=int(100*get_multiplier_factor(side_winrate,50))
        winrate_radiant=10000-multiplier_factor
        winrate_dire=multiplier_factor
        
        pick_radiant=list(map(int,x[1].pick_radiant[1:-1].split(',')))
        pick_dire=list(map(int,x[1].pick_dire[1:-1].split(',')))
    
    
        if x[1].win_radiant==True:
            val=create_data(pick_radiant,pick_dire,winrate_radiant)
        else:
            val=create_data(pick_dire,pick_radiant,winrate_dire)
    
        data_tour.extend(val)
    

    if remove_inf>0 and remove_post==False:
        data_tour=treat_data(data_tour,remove_inf)
        
    count_hero,nbr_hero=hero_set(data_tour)
    dico=create_hero_dico()
    name_score,beta=spring_rank_hero(data_tour,nbr_hero,dico)
    #
    if remove_inf>0 and remove_post==True:
        name_score=treat_data_post(name_score,data_tour,dico,remove_inf)
    # 
    return name_score,beta
        
def merge_name_score(list_name_score,list_beta):
# list_name_score -> list of the list obtained via create_hero_ranking
# list_beta -> list of the beta obtained via create_hero_ranking

# output a dataframe out of all the ranking, and combine so that they use the same beta (first beta of the list)
    df=pd.DataFrame(list_name_score[0],columns=['name',0])
    i=0
    for x in list_name_score[1:]:
        i+=1
        df=df.merge(pd.DataFrame(x,columns=['name',i]),left_on='name',right_on='name',how='outer')
        
    lenn=len(df.columns)
    for i in range(2,lenn):
        df[df.columns[i]]=link_ranking_different_beta(list_beta[df.columns[i]],df[df.columns[i]],list_beta[0])
        
    return df,list_beta[0]

def find_suitable_min(df):
# df -> datafram contaning data from different compute of spring rank from tournament data obtained from merge_name_score

# output non nan value from base column where the secondary column is as small as possible
    df=df[~df.isnull().any(axis=1)]
    if len(df)==0:
        return 0
    index_min=df.iloc[:,-1].idxmin()
    return df[0].loc[index_min]

def normalize_hero_dataframe(df):
# df -> dataframe containing data from different compute of spring rank from tournament data obtained from merge_name_score

# output dataframe with common minimum value for each col compared to the first one
    lenn=len(df.columns)
    for i in range(2,lenn):
        index_min=df[df.columns[i]].idxmin()
        # val_min=df[0].iloc[index_min]
        val_min=find_suitable_min(df[[0,df.columns[i]]])
        val_df=df[df.columns[i]].iloc[index_min]
        df[df.columns[i]]+=(val_min-val_df)
    return df

def count_hero_games(tournament,df):
# tournament -> path of tournament's data
# df -> dataframe of ranked heroes of the tournament (used in compare_inside_tournament)

# ouput df with an added column with number of time each heroes as been played
    data=pd.read_csv(tournament)
    df.insert(loc=1,column='number_game',value=0)
    dico=create_hero_dico()
    
    for x in data.iterrows():
        radiant=x[1].pick_radiant
        dire=x[1].pick_dire
        for y in radiant[1:-1].split(','):
            name=dico[int(y)]
            df.loc[df['name']==name,'number_game']+=1
        for y in dire[1:-1].split(','):
            name=dico[int(y)]
            df.loc[df['name']==name,'number_game']+=1
    return df
