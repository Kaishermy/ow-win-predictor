# Overview
This is a simple win predictor for the game Overwatch by Blizzard Entertainment. It uses a logistic regression model to train on local game data and predict the odds of winning an in-progress game. This is intended to serve as a demonstration of an application of supervised ML modelling. This is currently only confirmed to work under the following conditions:
- Windows 11
- 5v5 quick play
- 2560x1440p resolution
# Installation
```
> git clone https://github.com/Kaishermy/ow-win-predictor.git
> cd ow-win-predictor
> python overwatch_win_predictor.py
```
## Dependencies
- OpenCV
- pytesseract
- Sklearn
- PyAutoGUI
- Numpy
- Pandas
- pynput
# Usage
Upon starting the program, 5 options are presented. To select an option, simply type the corresponding letter and press enter. Some options offer additional options upon selection, for which the same process applies. An overview of each option follows.
## Add Training Data
Implements new training data into the dataset. This can be done in one of the two ways listed below. Regardless of method, the "B_Win" column (label) of the dataset file `ow_training_data.csv` will need to be manually populated. Label categories are binary: 0 for a red (enemy) team victory and 1 for a blue (friendly) team victory.
### Adding Data Via Files
To add data via files, fullscreen screenshots must be taken of post-match 5v5 scoreboards in the History tab of the user's Career Profile and added into the Training directory. The script will then extract the scoreboard statistics and add them to the dataset. 
### Random Population
This adds the given number of "dummy" rows to the dataset. Each of these rows are randomly populated with curated ranges. This is recommended for testing purposes only, and predicting win probability from random data is likely not going to produce accurate results.
## Initialize Dataset
Creates a blank dataset file, overriding the existing one if it exists. The new file will be called `ow_training_data.csv`.
## Train Model
Creates a model based on the training dataset that is used for future win predictions. This command must be executed before making any predictions. Models are saved across sessions in the file `model.joblib`.
## Predict New Vector
Predicts the probability of a blue (friendly) team victory based on the current on-screen scoreboard. Selecting this option starts a listener for either the backtick/grave (\`) or Delete keys. Pressing the latter exits the current state and return the user to the menu. Pressing the backtick key prompts the program into taking a screenshot, making a prediction, and deleting the screenshot. For this to work properly, the scoreboard must be open before and during a prediction while in-game.
## Quit
Exits the program.
