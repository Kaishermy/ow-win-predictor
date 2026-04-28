# Overview
This is a simple win predictor for the game Overwatch by Blizzard Entertainment. It uses a logistic regression model to train on local game data and predict the odds of winning an in-progress game.
## Compatibility
Currently this only works in 5v5 quick play.
# Installation
```
git clone https://github.com/Kaishermy/ow-win-predictor.git
python overwatch_win_predictor.py
```
## Dependencies
- OpenCV
- pytesseract
- Sklearn
- PyAutoGUI
- Numpy
- Pandas
- pynput
