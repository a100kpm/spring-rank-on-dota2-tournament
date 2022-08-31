# -*- coding: utf-8 -*-
"""
main
"""
from spring_rank import *
from get_data import *
from data_treatment import *


# dota_hero.csv can be obtained via opendota_api : https://www.opendota.com (data_treatment.py)
# get a key from valve api : https://cran.r-project.org/web/packages/CSGo/vignettes/auth.html (get_data.py)
# create_csv_from_data_tour(id_tournament)
# get_team_list()
# rank_team,beta_team=ranking_team()



# =============================================================================
# to add:
# ### impact of individual team on each heroes
# ### prediction post draft to add
# ### look at tendency between col of df (like taking account for team always decrease/increase value of X)
# =============================================================================



name_score1,beta1=create_hero_ranking(team_weight=False,side_weight=False,remove_inf=0,remove_post=False,team_list=[],path='tournament.csv')
name_score2,beta2=create_hero_ranking(team_weight=True,side_weight=False,remove_inf=0,remove_post=False,team_list=[],path='tournament.csv')
name_score3,beta3=create_hero_ranking(team_weight=False,side_weight=True,remove_inf=0,remove_post=False,team_list=[],path='tournament.csv')
name_score4,beta4=create_hero_ranking(team_weight=True,side_weight=True,remove_inf=0,remove_post=False,team_list=[],path='tournament.csv')

name_scorea,betaa=create_hero_ranking(team_weight=False,side_weight=False,remove_inf=14,remove_post=False,team_list=[],path='tournament.csv')
name_scorea2,betaa2=create_hero_ranking(team_weight=True,side_weight=False,remove_inf=14,remove_post=False,team_list=[],path='tournament.csv')
name_scorea3,betaa3=create_hero_ranking(team_weight=False,side_weight=True,remove_inf=14,remove_post=False,team_list=[],path='tournament.csv')
name_scorea4,betaa4=create_hero_ranking(team_weight=True,side_weight=True,remove_inf=14,remove_post=False,team_list=[],path='tournament.csv')

name_scoreb,betab=create_hero_ranking(team_weight=False,side_weight=False,remove_inf=14,remove_post=True,team_list=[],path='tournament.csv')
name_scoreb2,betab2=create_hero_ranking(team_weight=True,side_weight=False,remove_inf=14,remove_post=True,team_list=[],path='tournament.csv')
name_scoreb3,betab3=create_hero_ranking(team_weight=False,side_weight=True,remove_inf=14,remove_post=True,team_list=[],path='tournament.csv')
name_scoreb4,betab4=create_hero_ranking(team_weight=True,side_weight=True,remove_inf=14,remove_post=True,team_list=[],path='tournament.csv')

list_name_score=[name_score1,name_score2,name_score3,name_score4,
                     name_scorea,name_scorea2,name_scorea3,name_scorea4,
                     name_scoreb,name_scoreb2,name_scoreb3,name_scoreb4]

list_beta=[beta1,beta2,beta3,beta4,
           betaa,betaa2,betaa3,betaa4,
           betab,betab2,betab3,betab4]



df,beta=merge_name_score(list_name_score,list_beta)
df=normalize_hero_dataframe(df)
df2=df[~df.isnull().any(axis=1)] # no nan value


#see_gain(changement_val,beta)










