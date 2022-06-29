import numpy as np
import pandas as pd

from config import LEAGUE_CONFIG
from datetime import datetime as dt
from scipy.stats import poisson, skellam


def load_538_dataset():
    """
    Function to load football data using the fivethirtyeight api

    :return:
    df: The dataframe containing the fivethirtyeight data
    """

    # Set the url where the 538 football data is hosted
    data_url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'

    # Load in the data
    df = pd.read_csv(data_url)

    # Calculate the days since today for each game
    df['days_since'] = df.date.apply(lambda x: (dt.today() - dt.strptime(x, '%Y-%m-%d')).days)

    # Extract only the relevant columns
    df = df[['season',
             'date',
             'league',
             'team1',
             'team2',
             'score1',
             'score2',
             'xg1',
             'xg2',
             'nsxg1',
             'nsxg2',
             'days_since']]

    return df


def extract_season_data(df, league, year):
    """
    Function which extracts data for a given season from the
    fivethirtyeight data

    :param df: The dataframe containing the fivethirtyeight data
    :param league: The league to extract data for
    :param year: The year to extract data for

    :return:
    season_df: Extracted data for the season so far
    future_df: Extracted data for upcoming games in the season
    """

    # Extract the full league name from config
    league_title = LEAGUE_CONFIG[league]

    # Extract data only for required season
    season_df = df[df.league == league_title]
    season_df = season_df[season_df.season == year].copy()
    season_df = season_df.sort_values('days_since', ascending=False)

    # Get upcoming games for the next 7 days
    future_df = season_df[(season_df.days_since <= 0) &
                          (season_df.days_since > -7)]

    future_df = future_df[['season', 'date', 'team1', 'team2']]

    # Remove any games from season_df which have not occurred
    season_df = season_df.dropna()

    return season_df, future_df


def get_ratings(season_df, n_games=10):
    """
    Function which produces home advantage factors and attack / defence
    ratings for each team based on the most recent n_games

    :param season_df: The dataframe containing games for a given season
    :param n_games: The number of games to consider when calculating ratings

    :return:
    home_factor: A dict containing home advantage factor information
    recent_rating: A dataframe containing the team ratings
    """

    #############################################################
    # STEP 1 - Get overall attack/defence ratings for all teams #
    #############################################################

    # Work with new copy of season_df
    season_df = season_df.copy()

    # Calculate an aggregated score for each match, using G, xG & nsxG
    season_df['agg1'] = np.mean(season_df[['score1', 'xg1', 'nsxg1']], axis=1)
    season_df['agg2'] = np.mean(season_df[['score2', 'xg2', 'nsxg2']], axis=1)

    # Separate out home and away data
    home_df = season_df[['team1', 'agg1', 'agg2']]
    away_df = season_df[['team2', 'agg2', 'agg1']]

    # Rejoin the data
    full_cols = ['team', 'for', 'against']

    home_df.columns = full_cols
    away_df.columns = full_cols

    full_df = pd.concat([home_df, away_df])

    # Calculate the attack and defence strength of each team this season
    season_rating = full_df.groupby('team')['for', 'against'].mean().reset_index()

    ################################################
    # STEP 2 - Calculate the home advantage factor #
    ################################################

    # Calculate home, away & overall goal averages
    avgH = season_df.agg1.values.mean()
    avgA = season_df.agg2.values.mean()
    avg = (avgH + avgA) / 2

    home_factor = {'avgH': avgH, 'avgA': avgA, 'avg': avg}

    ###########################################################
    # STEP 3 - Calculate the fixture difficulty for each game #
    ###########################################################

    # Join season strength to season_df for fixture difficulty calculation
    season_df = season_df.merge(season_rating, left_on='team2', right_on='team')
    season_df.rename(columns={'for': 'att_diff1', 'against': 'def_diff1'}, inplace=True)
    season_df.drop('team', axis=1, inplace=True)

    season_df = season_df.merge(season_rating, left_on='team1', right_on='team')
    season_df.rename(columns={'for': 'att_diff2', 'against': 'def_diff2'}, inplace=True)
    season_df.drop('team', axis=1, inplace=True)

    # Account for home advantage in fixture difficulty
    season_df.att_diff1 *= (avgA / avg)
    season_df.def_diff1 *= (avgH / avg)
    season_df.att_diff2 *= (avgH / avg)
    season_df.def_diff2 *= (avgA / avg)

    #######################################################
    # STEP 4 - Calculate the recent performance each team #
    #######################################################

    # Separate out home and away data (again!)
    home_df = season_df[['team1', 'agg1', 'agg2', 'att_diff1', 'def_diff1', 'days_since']]
    away_df = season_df[['team2', 'agg2', 'agg1', 'att_diff2', 'def_diff2', 'days_since']]

    # Rejoin the data
    full_cols = ['team', 'for', 'against', 'att_diff', 'def_diff', 'days_since']

    home_df.columns = full_cols
    away_df.columns = full_cols

    full_df = pd.concat([home_df, away_df])

    # Only consider the more recent games
    full_df = full_df.sort_values('days_since', ascending=True)
    recent_df = full_df.groupby('team').head(n_games).reset_index()

    # Ensure that every team has the requisite number of games
    n_teams = len(season_rating)

    if len(recent_df) != n_games * n_teams:
        raise ValueError('Some teams do not have the requisite number of games')

    # Calculate the attack and defence strength of each team recently
    recent_rating = recent_df.groupby('team')['for', 'against', 'att_diff', 'def_diff'].mean().reset_index()

    recent_rating['att_rating'] = recent_rating['for'] / recent_rating['def_diff']
    recent_rating['def_rating'] = recent_rating['against'] / recent_rating['att_diff']

    recent_rating = recent_rating[['team', 'att_rating', 'def_rating']]

    return home_factor, recent_rating


def predict_games(fixture_df, home_factor, team_rating):
    """
    Function which produces home/draw/away and over/under 2.5 goal
    probabilities for a fixture list using home advantage and team ratings

    :param fixture_df: A dataframe containing upcoming fixtures to predict
    :param home_factor: A dict containing home advantage factor information
    :param team_rating: A dataframe containing the team ratings

    :return:
    fixture_df: The dataframe containing the probabilities
    """
    # Work with new copy of fixture_df
    fixture_df = fixture_df.copy()

    # Join team ratings onto fixture_df
    fixture_df = fixture_df.merge(team_rating, left_on='team1', right_on='team')
    fixture_df.rename(columns={'att_rating': 'att1', 'def_rating': 'def1'}, inplace=True)
    fixture_df.drop('team', axis=1, inplace=True)

    fixture_df = fixture_df.merge(team_rating, left_on='team2', right_on='team')
    fixture_df.rename(columns={'att_rating': 'att2', 'def_rating': 'def2'}, inplace=True)
    fixture_df.drop('team', axis=1, inplace=True)

    # Calculate the expected goals for each team
    fixture_df['exp1'] = fixture_df.att1 * fixture_df.def2 * home_factor['avgH']
    fixture_df['exp2'] = fixture_df.att2 * fixture_df.def1 * home_factor['avgA']

    # Calculate the match result probabilities using a skellam distribution
    skellam_rv = skellam(fixture_df.exp1, fixture_df.exp2)

    home_probs = 1 - skellam_rv.cdf(0)
    draw_probs = skellam_rv.pmf(0)
    away_probs = skellam_rv.cdf(-1)

    # Calculate the over/under 2.5 goals probability using a poisson distribution
    under_probs = poisson.cdf(2.5, fixture_df.exp1 + fixture_df.exp2)
    over_probs = 1 - under_probs

    # Append all this predictions
    fixture_df['home_prob'] = home_probs
    fixture_df['draw_prob'] = draw_probs
    fixture_df['away_prob'] = away_probs
    fixture_df['under_prob'] = under_probs
    fixture_df['over_prob'] = over_probs

    fixture_df.drop(['att1', 'def1', 'att2', 'def2'], axis=1, inplace=True)

    return fixture_df


def predict_upcoming_games_for_season(df, league, year):
    """
    Function which predicts upcoming games in a league

    :param df: The dataframe containing the fivethirtyeight data
    :param league: The league to produce predictions for
    :param year: The year to produce predictions for

    :return:
    predictions: A dataframe containing the predictions
    """

    # Get season data and upcoming fixtures
    season_df, future_df = extract_season_data(df, league, year)

    # Calculate home factor and team ratings
    home_factor, recent_rating = get_ratings(season_df)

    # Predict upcoming fixtures
    predictions = predict_games(future_df, home_factor, recent_rating)

    # Calculate fair odds
    predictions['home_fair_odds'] = round(1 / predictions.home_prob, 3)
    predictions['draw_fair_odds'] = round(1 / predictions.draw_prob, 3)
    predictions['away_fair_odds'] = round(1 / predictions.away_prob, 3)
    predictions['under_fair_odds'] = round(1 / predictions.under_prob, 3)
    predictions['over_fair_odds'] = round(1 / predictions.over_prob, 3)

    return predictions


if __name__ == '__main__':
    df = load_538_dataset()

    league_predictions = predict_upcoming_games_for_season(df, 'B1', 2022)
