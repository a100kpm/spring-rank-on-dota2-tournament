# -*- coding: utf-8 -*-
"""

"""
import csv
import urllib3
import json
import pandas as pd
from spring_rank import *


dota_mykey='******************************'
# get a key from valve api : https://cran.r-project.org/web/packages/CSGo/vignettes/auth.html

def get_dota_tournament_data_max_200(id_tn,mykey=dota_mykey):
# id_tn -> id of the tournament
# mykey -> key of valve's api

# output information about the tournament with format json (will only get first 200 results)
    http = urllib3.PoolManager()
    request_string='http://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V1/?key='+dota_mykey+'&league_id='+str(id_tn)
    req = http.request('GET',request_string)
    json_result=json.loads(req.data.decode('utf-8'))
    return json_result

def get_dota_tournament_data(id_tn,mykey=dota_mykey):
# id_tn -> id of the tournament
# mykey -> key of valve's api

# output information about the tournament with format json
    http = urllib3.PoolManager()
    request_string='http://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V1/?key='+dota_mykey+'&league_id='+str(id_tn)
    req = http.request('GET',request_string)
    json_result=json.loads(req.data.decode('utf-8'))
    if json_result['result']['results_remaining']>0:
        json_result_temp=json_result.copy()
        i=len(json_result['result']['matches'])
        
        while json_result_temp['result']['results_remaining']>0: 
            match_id=json_result_temp['result']['matches'][-1]['match_id']
            request_string=f"""http://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V1/?key={dota_mykey}&league_id={str(id_tn)}
            &start_at_match_id={str(match_id)}"""
            req = http.request('GET',request_string)
            json_result_temp=json.loads(req.data.decode('utf-8'))
            for value in json_result_temp['result']['matches']:
                i=len(json_result)
                if value in json_result['result']['matches']:
                    pass
                else:
                    json_result['result']['matches'].append(value)
      
    return json_result

def get_dota_game_data(game_id,mykey=dota_mykey):
# game_id -> id of game found in data from get_dota_tournament_data
# mykey -> key of valve's api

# output -> information about the game with format json
    http = urllib3.PoolManager()
    request_string='http://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/V1/?key='+dota_mykey+'&match_id='+str(game_id)
    req = http.request('GET',request_string)
    json_result=json.loads(req.data.decode('utf-8'))
    return json_result

def get_team_list(path='tournament.csv'):
#

# output the list of all team of the tournament
    data=pd.read_csv(path)
    return list(set(data.radiant) | set(data.dire))

def get_winrate_from_tour(data):
# data -> data from the the csv file 'tournament.csv' (created with create_csv_from_data_tour)

# output the winrate by side in the tournament
    winrate=data['win_radiant'].sum()/len(data)
    winrate=round(winrate*100,0)
    return int(winrate)

def get_multiplier_factor(winrate,spring_rank_factor):
# winrate -> winrate by side
# spring_rank_factor -> value created from generate_data_team_side_factor (data_treatment.py)

# output the multiplier taking both side and team into account for the creation of spring rank matrix A
    spring_rank_factor=100-spring_rank_factor
    val=winrate*(100-spring_rank_factor)/( winrate*(100-spring_rank_factor)+ (100-winrate)*spring_rank_factor)
    return 100*round(val,4)

def get_tournament_pick_data(id_tournament): 
# id_tournament -> tournament id

# output data for create_csv_from_data_tour
    a=get_dota_tournament_data(id_tournament)
    list_match=[]
    for x in a['result']['matches']:
        list_match.append(x['match_id']) 
    data_tour=[]
    
    for x in list_match:
        try:
            data_game=get_dota_game_data(x)
            pick_radiant=[]
            pick_dire=[]
            radiant_name=data_game['result']['radiant_name']
            dire_name=data_game['result']['dire_name']
            win=data_game['result']['radiant_win']
            
            for y in data_game['result']['picks_bans']:
                if y['is_pick']==False:
                    pass
                else:
                    if y['team']==0:
                        pick_radiant.append(y['hero_id'])
                    else:
                        pick_dire.append(y['hero_id'])
                        
            data=[radiant_name,dire_name,win]
            data.extend([pick_radiant])
            data.extend([pick_dire])
            
            data_tour.append(data)
        except KeyError:
            #bypath false match, like 1v1
            pass
    return data_tour

def create_csv_from_data_tour(id_tournament,path='tournament.csv'):
# id_tournament -> tournament id

# output a csv file with all the needed information and save it at path
    data_tour=get_tournament_pick_data(id_tournament)
    col_name=['radiant','dire','win_radiant','pick_radiant','pick_dire']
    with open(path, 'w',newline='') as f: 
        write = csv.writer(f) 
        write.writerow(col_name) 
        write.writerows(data_tour) 
        




