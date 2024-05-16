import pandas as pd
import numpy as np

STREAK = {}
STREAK_FUT = {}


def create_new_metrics(df: pd.DataFrame):
    """
    This function creates the following metrics for a Dataset:
        * Assist2TurnOverRatio: Number of Assists / Number of Turovers
        * OffensiveReboundRatio: Number of offensive rebounds / Number of total rebounds in ofensive field
        * FieldGoalPercentqage: Percentage of field goals made
        * PersonalFaultRatio: Percentage of personal faults during the game
    Then it will remove already used columns

    Input:
        df (pd.DataFrame): DataFrame containing RegularSeasonDetailedResults or NCAADetailedResults
    Output:
        df (pd.DataFrame): DataFrame with necessary columns
    """
    # Assists
    df['WA2TR'] = df['WAst'] / df['WTO']
    df['LA2TR'] = df['LAst'] / df['LTO']
    df['WA2TR-LA2TR'] = df['WA2TR'] - df['LA2TR']

    # Rebounds
    df['WOffR'] = df['WOR'] / (df['WOR'] + df['LDR'])
    df['LOffR'] = df['LOR'] / (df['LOR'] + df['WDR'])
    df['WOffR-LOffR'] = df['WOffR'] - df['LOffR']

    df['WAllR'] = (df['WOR'] + df['WDR']) / (df['WOR'] + df['WDR'] + df['LOR'] + df['LDR'])
    df['LAllR'] = 1 - df['WAllR']
    df['WAllR-LAllR'] = df['WAllR'] - df['LAllR']

    # Field goal
    df['WFGP'] = (df['WFGM'] + df['WFGM3']) / (df['WFGA'] + df['WFGA3'])
    df['LFGP'] = (df['LFGM'] + df['LFGM3']) / (df['LFGA'] + df['LFGA3'])

    # Personal Faults
    df['WPFR'] = df['WPF'] / (df['WPF'] + df['LPF'])
    df['LPFR'] = df['LPF'] / (df['WPF'] + df['LPF'])

    # Remove uninformative/unused columns
    used_columns = [
        'LFGA', 'LFGA3', 'WFGA', 'WFGA3', 'WFGM', 'WFGM3', 'LFGM3',
        'LFGM', 'WOR', 'WDR', 'LOR', 'LDR', 'WAst', 'LAst', 'WTO', 'LTO']

    uninformative_columns = [
        'WFTM', 'WFTA', 'LFTM', 'LFTA', 'WStl', 'LStl', 'WBlk', 'LBlk', 'NumOT',
        'WAllR-LAllR', 'WA2TR-LA2TR', 'WA2TR-LA2TR', 'WA2TR-LA2TR', 'WAllR-LAllR',
        'WOffR-LOffR', 'LScore', 'WScore']

    drop_columns = used_columns + uninformative_columns
    df.drop(columns=drop_columns, inplace=True)
    return df


def prepare_data_for_aggregation(df):
    """
    This function will prepare data for aggregation, converting it from W/L Teams to H/A Teams.

    Input:
        df (pd.DataFrame): Contains tha dataframe with new columns
    Output:
        df (pd.DataFrame): Refactors columns to some spatial data
    """
    # W/L to H/A
    df["Home"] = np.where(df["WLoc"] == "H", df["WTeamID"], df["LTeamID"])
    df["Away"] = np.where(df["WLoc"] != "H", df["WTeamID"], df["LTeamID"])
    df["Result"] = np.where(df["WLoc"] == "H", 1, 0)
    df['HPF'] = np.where(df["WLoc"] == "H", df['WPF'], df['LPF'])
    df['APF'] = np.where(df["WLoc"] != "H", df['WPF'], df['LPF'])

    df['HA2TR'] = np.where(df["WLoc"] == "H", df['WA2TR'], df['LA2TR'])
    df['AA2TR'] = np.where(df["WLoc"] != "H", df['WA2TR'], df['LA2TR'])

    df['HOffR'] = np.where(df["WLoc"] == "H", df['WOffR'], df['LOffR'])
    df['AOffR'] = np.where(df["WLoc"] != "H", df['WOffR'], df['LOffR'])

    df['HAllR'] = np.where(df["WLoc"] == "H", df['WAllR'], df['LAllR'])
    df['AAllR'] = np.where(df["WLoc"] != "H", df['WAllR'], df['LAllR'])

    df['HFGP'] = np.where(df["WLoc"] == "H", df['WFGP'], df['LFGP'])
    df['AFGP'] = np.where(df["WLoc"] != "H", df['WFGP'], df['LFGP'])

    df['HPFR'] = np.where(df["WLoc"] == "H", df['WPFR'], df['LPFR'])
    df['APFR'] = np.where(df["WLoc"] != "H", df['WPFR'], df['LPFR'])

    df.drop(
        columns=[
            'WTeamID', 'LTeamID', 'WPF', 'LPF', 'WA2TR', 'LA2TR', 'WOffR', 'LOffR',
            'LAllR', 'WAllR', 'WFGP', 'LFGP', 'WPFR', 'LPFR', 'WLoc'],
        inplace=True
    )
    return df


def compute_streak(hteam, ateam, result):
    """
    This function computes the consecutive wins/losses of two teams

    Input:
        hteam (str): ID of the home team
        ateam (str): ID of the away team
        result (int): Result of the game
    Output:
        (list): Contains the Streak prior the game
    """
    STREAK = STREAK_FUT.copy()
    hteam_streak = STREAK_FUT.get(hteam, 0)
    ateam_streak = STREAK_FUT.get(ateam, 0)
    if result == 1:
        hteam_streak = 1 if hteam_streak < 0 else hteam_streak + 1
        ateam_streak = -1 if ateam_streak > 0 else ateam_streak - 1
    else:
        ateam_streak = 1 if ateam_streak < 0 else ateam_streak + 1
        hteam_streak = -1 if hteam_streak > 0 else hteam_streak - 1

    STREAK_FUT[hteam] = hteam_streak
    STREAK_FUT[ateam] = ateam_streak
    return [STREAK.get(hteam, 0), STREAK.get(ateam, 0)]


def aggregate_data(df: pd.DataFrame):
    """
    This function aggregates the processed data as the last step before running the model.

    Input:
        df (pd.DataFrame): DataFrame with H/A headers
    Output:
        df (pd.DataFrame): DataFrme aggregated
    """
    home_columns = ['HPF', 'HA2TR', 'HOffR', 'HAllR', 'HFGP', 'HPFR']
    away_columns = ['APF', 'AA2TR', 'AOffR', 'AAllR', 'AFGP', 'APFR']
    new_column_names = ['PF', 'A2TR', 'OffR', 'AllR', 'FGP', 'PFR']
    dict_home = {x: v for x, v in zip(home_columns, new_column_names)}
    dict_away = {x: v for x, v in zip(away_columns, new_column_names)}
    final_df_columns = ['Season', 'DayNum', 'Home', 'Away', 'Result'] + home_columns + away_columns + ['HStreak', 'AStreak']
    final_df = pd.DataFrame(columns=final_df_columns)
    for idx, row in df.iterrows():
        home_home_last = df[(df['Season'] == row['Season']) & (df['DayNum'] < row['DayNum']) & (df['Home'] == row['Home'])]
        home_away_last = df[(df['Season'] == row['Season']) & (df['DayNum'] < row['DayNum']) & (df['Away'] == row['Home'])]

        away_home_last = df[(df['Season'] == row['Season']) & (df['DayNum'] < row['DayNum']) & (df['Home'] == row['Away'])]
        away_away_last = df[(df['Season'] == row['Season']) & (df['DayNum'] < row['DayNum']) & (df['Away'] == row['Away'])]

        home_home_last = home_home_last[home_columns]
        home_away_last = home_away_last[away_columns]

        away_home_last = away_home_last[home_columns]
        away_away_last = away_away_last[away_columns]

        streak = compute_streak(hteam=row['Home'], ateam=row['Away'], result=row['Result'])

        home_home_last = home_home_last.rename(columns=dict_home)
        home_away_last = home_home_last.rename(columns=dict_away)
        away_home_last = away_home_last.rename(columns=dict_home)
        away_away_last = away_away_last.rename(columns=dict_away)

        away_last = pd.concat([away_away_last, away_home_last])
        home_last = pd.concat([home_away_last, home_home_last])

        home = list(home_last.median())
        away = list(away_last.median())

        final_df.loc[len(final_df)] = list(row[['Season', 'DayNum', 'Home', 'Away', 'Result']].values) + home + away + streak
        print(f"{idx}/{len(df)}")

    return final_df


def preprae_regular_season_csv(file_name: str):
    """
    This function reads a CSV file and process the data for the model.

    Input:
        file_name (str): The file name of the CSV to process

    Output:
        df (pd.Dataframe): A pandas dataframe object containing the cleaned dataset
    """

    df = pd.read_csv(file_name)
    df = create_new_metrics(df=df)
    df = prepare_data_for_aggregation(df=df)
    df = aggregate_data(df=df)
    return df
