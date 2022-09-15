# -*- coding: utf-8 -*-
"""

"""
from data_treatment import *
import matplotlib.pyplot as plt
import os

def find_function(name=None):
# name -> name of the function to search; if left as None, will search for all function in directory

# output position of function in directory
    if name==None:
        for filename in os.listdir(os.getcwd()):
            if filename[-3:]=='.py':
                with open(os.path.join(os.getcwd(), filename), 'r') as f:
                    line_number=0
                    for line in f:
                        line_number+=1
                        if line[0:4]=='def ':
                            print(filename,'at line',line_number,' -->',line.split('(')[0][4:]+'()')

    else:
        for filename in os.listdir(os.getcwd()):
            if filename[-3:]=='.py':
                with open(os.path.join(os.getcwd(), filename), 'r') as f:
                    line_number=0
                    for line in f:
                        line_number+=1
                        if line[0:4]=='def ':
                            if name in line.split('(')[0][4:]:
                                print(f'function "{name}" is located in',filename,'at line',line_number,' -->',line[4:-2],'\n')
                                
                                
def compare_inside_tournament(tournament='tournament.csv',min_game=14):
# tournament -> path to the tournament's data
# min_game -> minimum number of game a hero need to have been played

# output 2 dataframe: df contains the rank of each heroes in the tournament depending on 12 way of using spring rank
#df2 is df without row contaning nan value
    list_name_score=[]
    list_beta=[]
    if min_game>0:
        name_score1,beta1=create_hero_ranking(team_weight=False,side_weight=False,remove_inf=0,remove_post=False,team_list=[],path=tournament)
        name_score2,beta2=create_hero_ranking(team_weight=True,side_weight=False,remove_inf=0,remove_post=False,team_list=[],path=tournament)
        name_score3,beta3=create_hero_ranking(team_weight=False,side_weight=True,remove_inf=0,remove_post=False,team_list=[],path=tournament)
        name_score4,beta4=create_hero_ranking(team_weight=True,side_weight=True,remove_inf=0,remove_post=False,team_list=[],path=tournament)
        
        
        name_scorea,betaa=create_hero_ranking(team_weight=False,side_weight=False,remove_inf=min_game,remove_post=False,team_list=[],path=tournament)
        name_scorea2,betaa2=create_hero_ranking(team_weight=True,side_weight=False,remove_inf=min_game,remove_post=False,team_list=[],path=tournament)
        name_scorea3,betaa3=create_hero_ranking(team_weight=False,side_weight=True,remove_inf=min_game,remove_post=False,team_list=[],path=tournament)
        name_scorea4,betaa4=create_hero_ranking(team_weight=True,side_weight=True,remove_inf=min_game,remove_post=False,team_list=[],path=tournament)
        list_name_score=[name_score1,name_score2,name_score3,name_score4,
                         name_scorea,name_scorea2,name_scorea3,name_scorea4]
        list_beta=[beta1,beta2,beta3,beta4,
                   betaa,betaa2,betaa3,betaa4]
        
    name_scoreb,betab=create_hero_ranking(team_weight=False,side_weight=False,remove_inf=min_game,remove_post=True,team_list=[],path=tournament)
    name_scoreb2,betab2=create_hero_ranking(team_weight=True,side_weight=False,remove_inf=min_game,remove_post=True,team_list=[],path=tournament)
    name_scoreb3,betab3=create_hero_ranking(team_weight=False,side_weight=True,remove_inf=min_game,remove_post=True,team_list=[],path=tournament)
    name_scoreb4,betab4=create_hero_ranking(team_weight=True,side_weight=True,remove_inf=min_game,remove_post=True,team_list=[],path=tournament)

    list_name_score.extend([name_scoreb,name_scoreb2,name_scoreb3,name_scoreb4])

    list_beta.extend([betab,betab2,betab3,betab4])
               

    df,beta=merge_name_score(list_name_score,list_beta)
    df=normalize_hero_dataframe(df)
    df=count_hero_games(tournament,df)
    df2=df[~df.isnull().any(axis=1)]
    
    return df,df2,beta

def compare_two_tournament(df,i,j):
# df -> dataframe contaning the heroes ranking of several tournament
# i and j -> col indices of the tournament to compare
    col_name='gain between '+str(i)+' and '+str(j)
    df[col_name]=df[j]-df[i]
# output dataframe with added column comparing gain/loss of each particular heroes
    return df

def compare_between_tournament(team_weight=True,side_weight=True,min_game=14,remove_post=True,
                               tournament=['tournament.csv','tournament2.csv'],dynamic_min_game=False):
# team_weight -> take into account the level of the team (should probably stay always on True)
# side_weight -> take into account the winrate of each side (should probably stay always on True)
# remove_inf -> remove heroes with less than min_game+1 match
# remove_post -> remove heroes before or after treatment
# tournament -> list of path toward file contaning data of each tournament to compare 
# dynamic_min_game -> if True, will choose a different min_game for each tournament so that roughly
#the same % of minimum pickrate is kept between each tournament
    
# output dataframe containing the ranking of each heroes for each tournament aswell as comparison on how
#each heroes did beter/worse on each subsequent tournament
    if type(tournament)!=list:
        print('need a list of path')
        return
    if len(tournament)<=1:
        print('need at least 2 tournament')
        return
    if dynamic_min_game==True:
        number_games=[]
        for x in tournament:
            number_games.append(len(pd.read_csv(x)))
        val=number_games[0]
        number_games=[math.ceil(min_game*x/val) for x in number_games]
            
            
    name_score,beta=create_hero_ranking(team_weight=team_weight,side_weight=side_weight,remove_inf=min_game,
                                   remove_post=remove_post,path=tournament[0])
    list_name_score=[name_score]
    list_beta=[beta]
    lenn=len(tournament)
    for i in range(1,lenn):
        if dynamic_min_game==True:
            min_game=number_games[i]
        name_score_temp,beta_temp=create_hero_ranking(team_weight=team_weight,side_weight=side_weight,remove_inf=min_game,
                                       remove_post=remove_post,path=tournament[i])
        list_name_score.append(name_score_temp)
        list_beta.append(beta_temp)
        
    df,beta=merge_name_score(list_name_score,list_beta)
    df=normalize_hero_dataframe(df)
    # can change here for col name (rename them before hand) instead of col num
    # will probably affect plot_hero_ranking_distribution_per_game
    lenn=len(list_name_score)-1
    for i in range(lenn):
        df=compare_two_tournament(df,i,i+1)
        
    return df,beta

def plot_hero_ranking_distribution_per_game(df):
# df -> dataframe contaning the heroes ranking of inside a tournament from create_hero_ranking

# output plots of heroes ranking in respect to their number of games for each way the ranking has been computed
# add an average and linear regression on each plot
    lenn=len(df.columns)
    max_val=df.iloc[:,2:].max().max()
    min_game=0
    max_game=df['number_game'].max()
    for i in range(lenn-2):
        df3=df[['number_game',i]].dropna()
        average=df3[i].mean()
        d=np.polyfit(df3['number_game'],df3[i],1)
        f=np.poly1d(d)
        
        ax=df3.plot(kind='scatter',x='number_game',y=i,ylim=(0,max_val),title=f'rank versus number of games')
        ax.axhline(average, color="red", linestyle="dashed")
        
        x1=min_game
        x2=max_game
        y1=f(min_game)
        y2=f(max_game)
        plt.plot([x1,x2],[y1,y2],color="green")

def plot_hero_ranking_between_tournament(df):
# df -> dataframe contaning the heroes ranking of several tournament

# output plots of the rank gain of each heroes on each subsequent tournaments
# add an average and linear regression on each plot
    lenn=int((len(df.columns)-2)/2)
    max_val=df.iloc[:,-lenn:].max().max()
    min_val=df.iloc[:,-lenn:].min().min()
    for i in range(lenn):
        col_name=[i,df.columns[-lenn+i]]
        df2=df[col_name]
        df2=df2.dropna()
        df2=df2.sort_values(col_name[0],ascending=False)
        df2.insert(0, 'rank_order', range(1,len(df2)+1))
        average=df2[col_name[-1]].mean()
        d=np.polyfit(df2[i],df2[col_name[-1]],1)
        f=np.poly1d(d)
        
        ax=df2.plot(kind='scatter',x=i,y=col_name[-1],ylim=(min_val,max_val),title=f'rank gain versus previous rank')
        ax.axhline(average, color="red", linestyle="dashed")
    
        min_rank=df2[i].min()
        max_rank=df2[i].max()
        x1=min_rank
        x2=max_rank
        y1=f(min_rank)
        y2=f(max_rank)
        plt.plot([x1,x2],[y1,y2],color="green")
        
def post_draft_prediction(data,rank_heroes,beta_heroes):
# data -> data of tournament's result (from csv file)
# rank_heroes -> list containing the spring rank ranking of each heroes (default parameter for best result)
# beta_heroes -> beta temperature associated from the above ranking 

# output list containing the result of matches and the prediction values of the result
    dico=create_hero_dico()
    winrate=get_winrate_from_tour(data)
    df=pd.DataFrame(rank_heroes)
    average=df[1].mean()
    win_lose=[]
    for x in data.iterrows():
        radiant=list( map(int,(x[1].pick_radiant[1:-1].split(', ')) ) )
        dire=list( map(int,(x[1].pick_dire[1:-1].split(', ')) ) )
        win=x[1].win_radiant
        
        radiant_score=[df[df[0]==dico[x]].iloc[0][1] if len(df[df[0]==dico[x]])>0 else average for x in radiant]
        dire_score=[df[df[0]==dico[x]].iloc[0][1] if len(df[df[0]==dico[x]])>0 else average for x in dire]
        
        lst_matchup=[proba_win(x,y,beta=beta_heroes)*100 for x in radiant_score for y in dire_score]
        lst_matchup=[get_multiplier_factor(winrate,x) for x in lst_matchup]
        
        win_predict=lst_matchup[0]
        for x in lst_matchup[1:]:
                win_predict=get_multiplier_factor(win_predict,x)
        win_lose.append([win,win_predict])
    win_lose=[[100,x[1]] if x[0]==True else [0,x[1]] for x in win_lose]
    return win_lose
    
def plot_error(win_lose):
# win_lose -> list containing the result of matches and the prediction values of the result

# output a plot of the predictions against the real result
# output a confusion matrix of the prediction
    df=pd.DataFrame(win_lose)
    df=df.rename(columns={0:'result',1:'prediction_value'})
    df['error']=abs(df['result']-df['prediction_value'])
    df=df.sort_values(['result','prediction_value'])
    df.insert(0, 'index', range(len(df)))
    
    ax=df.plot(kind='scatter',x='index',y='prediction_value',ylim=(0,100),title=f'predictions post draft compared to real results')
    ax.axhline(50, color="green", linestyle="dashed")
    df.plot(x='index',y='result',ax=ax,color='red',lw=3)
    
    
    conditions = [
    (df['result'] ==0),
    (df['result'] ==100) 
    ]
    values=['dire','radiant']

    df['result_name']=np.select(conditions,values)
    conditions= [
        (df['prediction_value']<50),
        (df['prediction_value']>=50)
        ]
    values=['dire','radiant']
    df['prediction_name']=np.select(conditions,values)

    confusion_matrix = pd.crosstab(df['result_name'], df['prediction_name'], rownames=['real_result'], colnames=['predicted_result'])
    return confusion_matrix

def plot_gain_with_team(df,team_name,beta):
# df -> dataframe containing impact of each team on the spring rank change of each heroes
# team_name -> name of the team
# beta -> beta temperature of the spring rank ranking

# output 2 graphs of the estimated spring rank score gain increase thanks to a particular team of heroes
#in respect to their spring rank ranking on the tournament
    df_temp=df[['heroes_name','global_rank',f'score_gain_with {team_name}','average_score_gain']]
    df_temp=df_temp[~df_temp.isnull().any(axis=1)]
    df_temp[f'score_gain_with {team_name}']=df_temp[f'score_gain_with {team_name}']-df_temp['average_score_gain']
    df_temp[f'spring_rank_gain_with {team_name}']=df_temp.apply(lambda x: 100*see_gain(x[f'score_gain_with {team_name}'],beta),axis=1)
    
    min_val=df_temp[f'score_gain_with {team_name}'].min()
    max_val=df_temp[f'score_gain_with {team_name}'].max()   
    
    d=np.polyfit(df_temp['global_rank'],df_temp[f'score_gain_with {team_name}'],1)
    f=np.poly1d(d)
    x1=df_temp['global_rank'].min()
    x2=df_temp['global_rank'].max()
    y1=f(x1)
    y2=f(x2)
    
    ax=df_temp.plot(kind='scatter',x='global_rank',y=f'score_gain_with {team_name}',ylim=(min_val,max_val),title=f'score gain with team {team_name}')
    plt.plot([x1,x2],[y1,y2],color="green")
    
    min_val=df_temp[f'spring_rank_gain_with {team_name}'].min()
    max_val=df_temp[f'spring_rank_gain_with {team_name}'].max()   
    
    d=np.polyfit(df_temp['global_rank'],df_temp[f'spring_rank_gain_with {team_name}'],1)
    f=np.poly1d(d)
    y1=f(x1)
    y2=f(x2)
    
    ax2=df_temp.plot(kind='scatter',x='global_rank',y=f'spring_rank_gain_with {team_name}',ylim=(min_val,max_val),
                     title=f'spring rank % gain with team {team_name}')
    plt.plot([x1,x2],[y1,y2],color="green")
    
def plot_gain_all_team(df,team_list,beta):
# df -> dataframe containing impact of each team on the spring rank change of each heroes
# team_list -> list containing all team of the tournament
# beta -> beta temperature of the spring rank ranking

# output 2 graphs for each teams of the estimated spring rank score gain increase thanks to a particular team of heroes
    for name in team_list:
        plot_gain_with_team(df,name,beta)
        
