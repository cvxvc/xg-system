# xG System

## Overview

Within this repo is a model to predict football results, found in `model.py`. It works by looking at the goals, xG and nsxG for a team in their recent games, and then compares this to how an average team would have performed in those fixtures. This is then used to compute attack and defence ratings for each team, which can be used to predict future games.

Also within this repo is `backtest_system.py` which analyses how the model performs on historical odds data. A simple betting system using the model appears to be profitable with 7% ROI when run on 8 leagues over a 3 year period.
