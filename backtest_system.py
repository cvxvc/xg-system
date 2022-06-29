import model as mdl
import numpy as np
import pandas as pd

from datetime import datetime as dt
from fuzzywuzzy import fuzz
from scipy.optimize import linear_sum_assignment


def produce_backtest_model_probs(season_df):
    """
    Produce the model probabilities for what they would have been before
    every particular match in a season

    :param season_df: The dataframe containing games for a given season

    :return:
    model_df: The dataframe containing the model predictions
    """

    # Get all the distinct dates where games were played
    dates = sorted(list(season_df.date.unique()))

    # Initialise an empty list which will contain model predictions
    model_df = []

    # Loop through each individual date
    for _, date in enumerate(dates):
        # Get the fixtures being played on that individual date
        fixture_df = season_df[season_df.date == date]

        # Get the fixtures played prior to this date
        prior_df = season_df[season_df.date < date]

        # Attempt to get ratings from the prior_df
        try:
            home_factor, recent_rating = mdl.get_ratings(prior_df)
        except ValueError:
            # Skip this date if unable to get ratings due to insufficient games played
            continue

        # Produce the predictions
        predictions = mdl.predict_games(fixture_df, home_factor, recent_rating)

        # Keep only the relevant columns
        predictions = predictions[['date',
                                   'team1',
                                   'team2',
                                   'score1',
                                   'score2',
                                   'home_prob',
                                   'draw_prob',
                                   'away_prob']]

        # Append the predictions to the model_df list
        model_df.append(predictions)

    # Concatenate all the predictions into one dataframe
    model_df = pd.concat(model_df)

    return model_df


def load_odds_data(league, year):
    """
    Function to load historical odds data from football-data.co.uk
    for backtesting the model

    :param league: The league to extract data for
    :param year: The year to extract data for

    :return:
    odds_df: The dataframe containing the odds data
    """

    # Construct the url where the data is hosted
    season = int(str(year)[-2:])
    season = str(season) + str(season + 1)
    data_url = 'https://www.football-data.co.uk/mmz4281/{}/{}.csv'.format(season, league)

    # Load in the data
    odds_df = pd.read_csv(data_url, encoding='latin')

    # Only keep the relevant columns
    odds_df = odds_df[['Date',
                       'HomeTeam',
                       'AwayTeam',
                       'FTHG',
                       'FTAG',
                       'MaxH',
                       'MaxD',
                       'MaxA']]

    # Rename the columns
    odds_df.columns = ['date',
                       'team1',
                       'team2',
                       'score1',
                       'score2',
                       'home_odds',
                       'draw_odds',
                       'away_odds']

    # Covert the date column to a string
    try:
        odds_df.date = odds_df.date.apply(lambda x: dt.strptime(x, '%d/%m/%Y'))
        odds_df.date = odds_df.date.apply(lambda x: dt.strftime(x, '%Y-%m-%d'))
    except ValueError:
        odds_df.date = odds_df.date.apply(lambda x: dt.strptime(x, '%d/%m/%y'))
        odds_df.date = odds_df.date.apply(lambda x: dt.strftime(x, '%Y-%m-%d'))

    return odds_df


def match_team_names(list_x, list_y):
    """
    Function to pair team names in one list to another, where the names
    may be slightly different in each list

    :param list_x: The first list containing the team names
    :param list_y: The second list containing the team names

    :return:
    matching: A dictionary mapping team names in list_x to list_y
    """

    # Ensure the two lists have the same length
    if len(list_x) != len(list_y):
        raise ValueError('Received lists of unequal length')

    # Use fuzzy logic to calculate a similarity rating for each pair of names
    similarity = [[fuzz.partial_ratio(x, y) for x in list_x] for y in list_y]

    # Cast this data into a numpy array (matrix)
    similarity_matrix = np.array(similarity)

    # Use a matching algorithm to determine the optimal matching between the two lists
    _, col_ind = linear_sum_assignment(-similarity_matrix)

    # Re-order list_x to match list_y
    list_x = [list_x[x] for x in col_ind]

    # Get the matching into a dictionary
    matching = dict(zip(list_x, list_y))

    return matching


def merge_model_and_odds_data(model_df, odds_df):
    """
    This function merges the back-tested model probabilities
    to the historical odds data

    :param model_df: The back-tested model probabilities
    :param odds_df: The historical odds data

    :return:
    merged_df: The dataframe containing the merged data
    """

    # Get the team names for both dataframes
    model_team_names = sorted(list(set(list(model_df.team1) + list(model_df.team2))))
    odds_team_names = sorted(list(set(list(odds_df.team1) + list(odds_df.team2))))

    # Match the team names
    matching_dict = match_team_names(odds_team_names, model_team_names)

    # Take a copy of the odds dataframe
    merged_df = odds_df.copy()

    # Change the team names so that they can be merged onto the model_df
    merged_df.team1 = merged_df.team1.apply(lambda x: matching_dict[x])
    merged_df.team2 = merged_df.team2.apply(lambda x: matching_dict[x])

    # Finally merge the data
    merged_df = merged_df.merge(model_df,
                                on=['team1', 'team2', 'date', 'score1', 'score2'],
                                how='left')

    return merged_df


def perform_betting_strategy(merged_df, threshold=0.2):
    """
    A very simple betting strategy which can be run on the merged model and odds
    data to backtest the algorithm

    :param merged_df: The dataframe containing the merged data
    :param threshold: The threshold in EV to use for determining if a bet should be placed

    :return:
    bet_df: The dataframe containing information about bets made using the strategy
    """

    # Take a copy of the merged_df
    bet_df = merged_df.copy()

    # Calculate the expected value of a unit bet on the home / away outcomes
    bet_df['home_value'] = bet_df.home_prob * bet_df.home_odds - 1
    bet_df['away_value'] = bet_df.away_prob * bet_df.away_odds - 1

    # Determine if a bet should be made using the threshold
    bet_df['home_bet'] = (bet_df.home_value > threshold).astype(int)
    bet_df['away_bet'] = (bet_df.away_value > threshold).astype(int)

    # Determine the actual result of the match
    bet_df['home_win'] = (bet_df.score1 > bet_df.score2).astype(int)
    bet_df['away_win'] = (bet_df.score2 > bet_df.score1).astype(int)

    # Calculate the profit of the bet for each outcome
    bet_df['home_profit'] = bet_df.home_bet * (bet_df.home_win * bet_df.home_odds - 1)
    bet_df['away_profit'] = bet_df.away_bet * (bet_df.away_win * bet_df.away_odds - 1)

    # Calculate the total profit
    bet_df['profit'] = bet_df.home_profit + bet_df.away_profit

    # Drop un-needed columns
    bet_df = bet_df.drop(['home_value',
                          'away_value',
                          'home_win',
                          'away_win',
                          'home_profit',
                          'away_profit'],
                         axis=1)

    return bet_df


def backtest_numerous_leagues(df):
    """
    Function to backtest the model over a number of leagues and years, using
    the simple betting strategy defined in perform_betting_strategy

    :param df: The dataframe containing the fivethirtyeight data

    :return:
    backtest_df: The dataframe containing the results of the backtest
    """

    # Define the leagues and years to backtest over
    leagues = ['E0', 'E1', 'F1', 'D1', 'D2', 'I1', 'SP1', 'P1']
    years = [2019, 2020, 2021]

    # Initialise an empty list which will contain the backtest results
    backtest_df = []

    # Loop over the previously defined leagues and years
    for league in leagues:
        for year in years:
            print('\nProcessing {} {}\n'.format(league, year))

            # Extract the season_df
            season_df, _ = mdl.extract_season_data(df, league, year)

            # Load the odds data
            odds_df = load_odds_data(league, year)

            # Compute the retrospective model probabilities
            model_df = produce_backtest_model_probs(season_df)

            # Merge the odds_df and model_df
            merged_df = merge_model_and_odds_data(model_df, odds_df)

            # Use the simple betting strategy on the data
            bet_df = perform_betting_strategy(merged_df)

            # Set the league and year in the dataframe
            bet_df['league'] = league
            bet_df['year'] = year

            # Append the predictions to the backtest_df list
            backtest_df.append(bet_df)

    # Concatenate the backtest data into one dataframe
    backtest_df = pd.concat(backtest_df)

    # Remove any rows where a bet was not placed
    backtest_df = backtest_df[backtest_df.profit != 0]

    # Drop and rows with null values
    backtest_df = backtest_df.dropna()

    return backtest_df


if __name__ == '__main__':
    df = mdl.load_538_dataset()

    backtest_df = backtest_numerous_leagues(df)

    backtest_roi = round(100 * backtest_df.profit.sum() / len(backtest_df), 3)

    print('Backtest ROI = {} %'.format(backtest_roi))

    backtest_df = backtest_df.sort_values('date')
