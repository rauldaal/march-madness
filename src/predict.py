import pandas as pd

REGION_MATCH_UP = {
    "W": "X",
    "Y": "Z"
}


def compute_ncaa_bracket(seed: pd.DataFrame):
    seed = seed[seed['Tournament'] == 'M']
    seed['Conference'] = seed['Seed'].str.slice(0, 1)
    seed['Position'] = seed['Seed'].str.slice(-2).astype(int)

    df = pd.DataFrame(columns=['Conference', 'Game', 'TeamA', 'TeamB'])
    new_order = []
    for conference in list(seed['Conference'].unique()):
        conf = seed[seed['Conference'] == conference]
        while not conf.empty:
            top_team = conf[conf['Position'] == conf['Position'].min()]
            low_team = conf[conf['Position'] == conf['Position'].max()]

            game = top_team['Seed'].values[0][1:] + low_team['Seed'].values[0][1:]

            df.loc[len(df)] = [conference, game, top_team['TeamID'].values[0], low_team['TeamID'].values[0]]

            conf = conf.drop(top_team.index, inplace=False, axis='index')
            conf = conf.drop(low_team.index, inplace=False, axis='index')

        df_conf = df[df['Conference'] == conference]
        games = list(df_conf.index)
        splits_top = [min(games[0: len(games) // 2]), max(games[0: len(games) // 2]), min(games[len(games) // 2:]), max(games[len(games) // 2:])]
        for i in splits_top:
            games.remove(i)
        split_bottom = games.copy()
        # create order for split_top
        splits_top.reverse()
        splits_top.insert(0, splits_top[-1])
        splits_top.pop()
        # Create order for split_bottom
        split_bottom.reverse()
        split_bottom.insert(-1, split_bottom[0])
        del split_bottom[0]
        splits_top.extend(split_bottom)
        new_order.extend(splits_top)

    df = df.reindex(new_order)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    next_df = df.copy()
    final_df = df.copy()

    while not next_df.empty and len(next_df) > 1:
        df = pd.DataFrame(columns=['Conference', 'Game', 'TeamA', 'TeamB'])
        for conference in list(next_df['Conference'].unique()):
            conf = next_df[next_df['Conference'] == conference]

            if len(conf) > 1:
                for i in range(0, len(conf), 2):
                    top_team = conf.iloc[i]
                    low_team = conf.iloc[i + 1]
                    game = top_team['Game'] + "_" + low_team['Game']
                    final_df.loc[len(final_df)] = [conference, game, top_team['Game'], low_team['Game']]
                    df.loc[len(df)] = [conference, game, top_team['Game'], low_team['Game']]

        next_df = df.copy()

    return final_df


def pregame_data(teamA, teamB, data, columns):

    home_columns = ['Home', 'HPF', 'HA2TR', 'HOffR', 'HAllR', 'HFGP', 'HPFR', 'HStreak']
    away_columns = ['Away', 'APF', 'AA2TR', 'AOffR', 'AAllR', 'AFGP', 'APFR', 'AStreak']
    # Cogemos el ultimmo partido de ambos equipos
    teamA_data = data[(data['Season'] == 2024) & ((data['Home'] == teamA) | (data['Away'] == teamA))].tail(1)
    teamB_data = data[(data['Season'] == 2024) & ((data['Home'] == teamB) | (data['Away'] == teamB))].tail(1)

    if teamA_data['Home'].values[0] == teamA:
        teamA_data = teamA_data[home_columns]
    else:
        teamA_data = teamA_data[away_columns]
        renamed_a2h_columns = {x: v for x, v in zip(away_columns, home_columns)}
        teamA_data = teamA_data.rename(columns=renamed_a2h_columns)

    if teamB_data['Home'].values[0] == teamB:
        teamB_data = teamB_data[home_columns]
        renamed_h2a_columns = {x: v for x, v in zip(home_columns, away_columns)}
        teamB_data = teamB_data.rename(columns=renamed_h2a_columns)
    else:
        teamB_data = teamB_data[away_columns]
    teamA_data.reset_index(inplace=True, drop=True)
    teamB_data.reset_index(inplace=True, drop=True)
    game = pd.concat([teamA_data, teamB_data], axis='columns')
    game['Season'] = 2024
    game['DayNum'] = 137
    game['Result'] = None

    return game[columns]


def generate_prediction(game_data, model):
    game_data.drop(columns=['Result'], inplace=True)
    result = model.predict(game_data)
    if result == 1:
        return game_data['Home'].values[0]
    else:
        return game_data['Away'].values[0]


def predict_ncaa_bracket(bracket, season_data, model):
    for i in range(len(bracket)):
        game = bracket.loc[i, :]
        teamA, teamB, game_name, conference = game['TeamA'], game['TeamB'], game['Game'], game['Conference']
        print(f"Game: - {game_name}     Teams: {teamA} vs {teamB}")
        game_data = pregame_data(teamA=teamA, teamB=teamB, data=season_data, columns=season_data.columns)
        winner = generate_prediction(game_data=game_data, model=model)
        print(f"Game: - {game_name}     Teams: {teamA} vs {teamB}  --> Winner  {winner}")
        # Update Winner
        idx = bracket[(bracket['Conference'] == conference) & (bracket['TeamA'] == game_name)].index.values
        if len(idx) > 0:
            idx = idx[0]
            bracket.at[idx, 'TeamA'] = winner

        idx = bracket[(bracket['Conference'] == conference) & (bracket['TeamB'] == game_name)].index.values
        if len(idx) > 0:
            idx = idx[0]
            bracket.at[idx, 'TeamB'] = winner
        bracket.loc[(bracket['Conference'] == conference) & (bracket['Game'] == game_name), "Winner"] = winner

    return bracket


def add_teamsName_2_bracket_output(bracke_df, teams):
    merge = bracke_df.merge(teams, left_on='TeamA', right_on='TeamID')
    merge["TeamAName"] = merge["TeamName"]
    merge.drop(columns=["TeamID", "TeamName", "FirstD1Season", "LastD1Season"], inplace=True)
    merge = merge.merge(teams, left_on='TeamB', right_on='TeamID')
    merge["TeamBName"] = merge["TeamName"]
    merge.drop(columns=["TeamID", "TeamName", "FirstD1Season", "LastD1Season"], inplace=True)
    merge['level'] = merge['Game'].str.count('_')
    merge['level'] = merge['level'].apply(lambda x: list(merge['level'].unique()).index(x))

    return merge


def predict_final_four(final_four_data, model):
    for k,v in REGION_MATCH_UP.items():
        winner_k, game_id_k = final_four_data[final_four_data['Conference'] == k][["Winner", "Game"]].values[0]
        winner_v, game_id_v = final_four_data[final_four_data['Conference'] == k][["Winner", "Game"]].values[0]
        final_four_data.loc[len(final_four_data)] = [f"{k}_{v}", f"{game_id_k}_{game_id_v}", winner_k, winner_v, None]




def predict(seed_file: str, season_data: pd.DataFrame, model):
    seed = pd.read_csv(seed_file)
    bracket = compute_ncaa_bracket(seed)
    solved_bracket = predict_ncaa_bracket(bracket=bracket, season_data=season_data, model=model)
    final_four = solved_bracket.tail(4)
    teams = pd.read_csv('data/MTeams.csv')

    merge = add_teamsName_2_bracket_output(solved_bracket, teams)
    return merge
