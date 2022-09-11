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
# create_csv_from_data_tour(id_tournament)
# get_team_list()
# rank_team,beta_team=ranking_team()



# =============================================================================
# to add:
# ### impact of individual team on each heroes
# =============================================================================



# df1,_,beta1=compare_inside_tournament('tournament.csv')
# plot_hero_ranking_distribution_per_game(df1)


# df2,beta2=compare_between_tournament(tournament=['tournament.csv','tournament2.csv'],dynamic_min_game=False)
# plot_hero_ranking_between_tournament(df2)
    

# path='tournament.csv'
# data=pd.read_csv(path)
# rank_heroes,beta_heroes=create_hero_ranking(path=path)
# win_lose=post_draft_prediction(data,rank_heroes,beta_heroes)
# print(plot_error(win_lose))
