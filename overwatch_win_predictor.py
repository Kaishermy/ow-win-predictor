import os
import pyautogui
import cv2
import pytesseract
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pynput.keyboard import Key, Listener, KeyCode
import msvcrt

column_names = ['B_Eliminations', 'B_Assists', 'B_Deaths', 'B_Damage', 'B_Healing', 'B_Mitigated',
                'R_Eliminations', 'R_Assists', 'R_Deaths', 'R_Damage', 'R_Healing', 'R_Mitigated']


def create_vector(img_file, left_offset, top_offset):
    """
    Create a vector by summing the given image's stats column-by-column.
    :param img_file: The current image file path being examined
    :param left_offset: The distance (pixels) between the left edge of the screen and left edge of the crop
    :param top_offset: The distance (pixels) between the top of the screen and the top of the crop
    :return: A vectorized form of the scoreboard being scanned, where each column is the sum of that stat
    """

    new_vector = []
    pixels_to_jump = 75
    width = 75

    for _ in range(2):  # 2 regions
        for _ in range(3):  # 3 stats per region (due to horizontal gap between E/A/D and DMG/H/MIT)
            image = cv2.imread(img_file)
            image = image[top_offset:top_offset + 400, left_offset:left_offset + width]
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert from BGR to grayscale
            cfg = r'--psm 6 -c tessedit_char_whitelist=0123456789'
            extracted_text = pytesseract.image_to_string(image_gray, config=cfg)
            stats = extracted_text.split()

            # Convert each stat to an integer and add to the current row
            stats = [int(stat) for stat in stats]
            new_vector.append(sum(stats))

            left_offset += pixels_to_jump

        left_offset += 260
        pixels_to_jump = 125
        width = 100

    return new_vector


def get_df():
    """
    Attempt to get the dataset from file.
    :return: Pandas df if successful and None otherwise
    """

    training_data_path = "ow_training_data.csv"
    if os.path.exists(training_data_path):
        return pd.read_csv("ow_training_data.csv")
    else:
        print(training_data_path, "does not exist")
        return None


def add_data():
    """
    Open the images one-by-one and extract data.
    """

    # Initialize and check if df exists, returning if not
    if (df := get_df()) is None:
        return

    training_dir = r"Training"
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
        return

    for file in os.scandir(training_dir):
        vector_blue = create_vector(file.path, 1215, 340)  # Blue team stats
        vector_red = create_vector(file.path, 1215, 900)  # Red team stats
        df.loc[len(df)] = vector_blue + vector_red + [-1]

    df.to_csv("ow_training_data.csv", index=False)


def add_rand_data(num_rows):
    """
    Add num_rows rows of random vectors to the dataset
    """

    # Initialize and check if df exists, returning if not
    if (df := get_df()) is None:
        return

    rng = np.random.default_rng()
    for _ in range(num_rows):
        new_vector = []
        for _ in range(2):  # Get blue and red stats
            new_vector += list(rng.integers(low=0, high=75, size=3))
            new_vector += list(rng.integers(low=0, high=100000, size=3))

        df.loc[len(df)] = new_vector + [-1]  # -1 is a placeholder for win value

    df.to_csv("ow_training_data.csv", index=False)


def train_data():
    """
    Train a linear regression model based on the existing dataset
    :return: The newly trained linear regression model
    """
    if (df := get_df()) is None:
        return

    training_data, testing_data = train_test_split(df, test_size=0.3)
    # print(f"\nTraining Data:\n{training_data}")
    # print(f"\nTesting Data:\n{testing_data}")

    training_labels = training_data["B_Win"]
    training_data = training_data.drop(["B_Win"], axis=1)
    testing_labels = testing_data["B_Win"]
    testing_data = testing_data.drop(["B_Win"], axis=1)
    # print(f"\nTraining Labels:\n{training_labels}")
    # print(f"\nTesting Labels:\n{testing_labels}")

    my_lr = LogisticRegression()
    my_lr_model = my_lr.fit(training_data, training_labels)
    my_model_predictions = my_lr_model.predict(testing_data)
    # print(f"\nPredictions:\n{my_model_predictions}")
    # print(f"\nActual Labels:\n{testing_labels}")

    my_cm = confusion_matrix(testing_labels, my_model_predictions)
    # print(f"\nConfusion Matrix:\n{my_cm}")
    # print(f"\nOriginal Probabilities:\n{my_lr_model.predict_proba(testing_data)}")

    return my_lr_model


def predict_vector_new(prediction_model, key):
    img_path = "predict.png"
    new_vector = []

    if key == KeyCode.from_char('`'):
        print("Taking screenshot...")
        screenshot = pyautogui.screenshot(img_path)

        vector_blue = create_vector(img_path, 880, 280)  # Blue team stats
        vector_red = create_vector(img_path, 880, 805)  # Red team stats
        new_vector = vector_blue + vector_red

        df = pd.DataFrame([new_vector], columns=column_names)
        prediction = prediction_model.predict(df)
        win_chance = prediction_model.predict_proba(df)[0][1] * 100
        predicted_outcome = "You will win!" if prediction == 1 else "You will lose..."
        print(f"\n{predicted_outcome} ({win_chance:.2f}% chance of winning)")

    if key == Key.delete:
        return False


def listen_for_key(my_lr_model):
    print("\nPress ` to take a screenshot and make a prediction.")
    print("Press Delete to exit.")

    with Listener(on_press=lambda event: predict_vector_new(my_lr_model, event)) as listener:
        listener.join()

    # Flush stdin so we don't dump entered chars from buffer during next input
    while msvcrt.kbhit():
        msvcrt.getch()


def predict_vector(my_lr_model):
    """
    Predict the label category for a newly entered data vector
    :param my_lr_model: The linear regression model
    """
    new_vector = []

    print()
    for curr_col in column_names:
        while True:
            try:
                new_value = int(input(f"Enter a value for {curr_col}: "))
            except ValueError:
                continue
            break

        new_vector.append(int(new_value))

    df = pd.DataFrame([new_vector], columns=column_names)

    prediction = my_lr_model.predict(df)
    win_chance = my_lr_model.predict_proba(df)[0][1] * 100
    predicted_outcome = "You will win!" if prediction == 1 else "You will lose..."
    print(f"\n{predicted_outcome} ({win_chance}% chance)")


def intro():
    """
    Prompt for user input repeatedly and call relevant helper functions until quitting
    """
    model = None  # Stores the logistic regression model if it exists

    print("\nWelcome to the Overwatch win predictor by Kai Sherman!")

    while True:
        print("\nAdd training data (a)\n"
              "Initialize dataset (i)\n"
              "Train model (t)\n"
              "Predict new vector (p)\n"
              "Quit (q)")
        choice = input("Enter your choice: ")

        if choice.lower() == 'a':
            choice_2 = input("Add data from file (a) or random data (r)? ")
            if choice_2.lower() == 'a':
                add_data()
            elif choice_2.lower() == 'r':
                num = input("Enter the number of random rows to add: ")
                add_rand_data(int(num))

        elif choice.lower() == 'i':
            confirmation = input("This will delete all data and reset the dataset! Are you sure (y/n)? ")
            if confirmation.lower() == 'n':
                continue
            # TODO: Something

        elif choice.lower() == 't':
            model = train_data()
            print("Training successful!")

        elif choice.lower() == 'p':
            if model is None:
                print("Model does not exist. Make sure to train the model first!")
                continue
            listen_for_key(model)

        elif choice.lower() == 'q':
            break
        else:
            print("Invalid input.")


def main():
    intro()


if __name__ == '__main__':
    main()
