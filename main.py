# -*- coding: utf-8 -*-
"""
main
"""
from spring_rank import *
from get_data import *
from data_treatment import *
from visual import *


# dota_hero.csv can be obtained via opendota_api : https://www.opendota.com (data_treatment.py)
# get a key from valve api : https://cran.r-project.org/web/packages/CSGo/vignettes/auth.html (get_data.py)
find_function('find_function')


# =============================================================================
# getting started
# # create_csv_from_data_tour(id_tournament,path='tournament.csv')
# ----> will generate data needed to run bellow function
# ----> you will need to find the id of a tournement you want to investigate on
# ----> you will need to modify dota_mykey value with your own valve api key
# uncomment bellow to try some of the visual of the project
# =============================================================================


# =============================================================================
# # First plot
# df1,_,beta1=compare_inside_tournament('tournament.csv')
# plot_hero_ranking_distribution_per_game(df1)
# =============================================================================



# =============================================================================
# # # Second plot
# # df2,beta2=compare_between_tournament(tournament=['tournament.csv','tournament2.csv'],dynamic_min_game=False)
# # plot_hero_ranking_between_tournament(df2)
# =============================================================================



# =============================================================================
# # # Third plot
# # # Warning the computation of hero_rank_gain_per_team() can be very long as it will
# # # easily require to compute the spring rank ranking over 500 times. This can take 
# # # over 1h to compute
# # df_global,beta,team_list=hero_rank_gain_per_team(path='tournament.csv')
# # plot_gain_all_team(df_global,team_list,beta)
# =============================================================================


# =============================================================================
# team ranking
# ---> 
# rank,beta=ranking_team(path='tournament.csv',start_beta=0,end_beta=10)
# print(proba_win(rank[0][1],rank[1][1],beta))
# =============================================================================
