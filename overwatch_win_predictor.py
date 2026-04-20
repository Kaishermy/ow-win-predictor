import os
import pyautogui
import cv2
import pytesseract
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def create_vector(img_file, left_offset, width, pixels_to_jump):
    """
    Create a vector by summing the given image's stats column-by-column.
    :param img_file: The current image file being examined
    :param left_offset: The starting point from the left screen edge (in pixels)
    :param width: The width of each crop
    :param pixels_to_jump: The number of pixels to shift left_offset each loop
    :return new_vector: A vectorized form of the scoreboard being scanned, where each column is the sum of that stat
    """

    new_vector = []

    for _ in range(3):  # 3 stats per region
        image = cv2.imread(img_file.path)
        image = image[340:340 + 400, left_offset:left_offset + width]
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert from BGR to grayscale
        cfg = r'--psm 6 -c tessedit_char_whitelist=0123456789'
        extracted_text = pytesseract.image_to_string(image_gray, config=cfg)
        stats = extracted_text.split()

        # Convert each stat to an integer
        stats = [int(stat) for stat in stats]
        new_vector.append(sum(stats))

        left_offset += pixels_to_jump

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
    :return:
    """

    if (df := get_df()) is None:
        return

    training_dir = r"Training"
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
        return

    for file in os.scandir(training_dir):
        vector_a = create_vector(file, 1215, 75, 75)
        vector_b = create_vector(file, 1475, 100, 125)
        df.loc[len(df)] = vector_a + vector_b + [-1]

    df.to_csv("ow_training_data.csv", index=False)


def add_rand_data(num_rows):
    """
    Add num_rows rows of random vectors to the dataset
    :return:
    """

    if (df := get_df()) is None:
        return

    rng = np.random.default_rng()
    for _ in range(num_rows):
        vector_a = list(rng.integers(low=0, high=75, size=3))
        vector_b = list(rng.integers(low=0, high=100000, size=3))
        df.loc[len(df)] = vector_a + vector_b + [-1]

    df.to_csv("ow_training_data.csv", index=False)


def train_data():
    """
    Train a linear regression model based on the existing dataset
    :return my_lr_model: The newly trained linear regression model
    """
    if (df := get_df()) is None:
        return

    training_data, testing_data = train_test_split(df, test_size=0.3)
    print(f"\nTraining Data:\n{training_data}")
    print(f"\nTesting Data:\n{testing_data}")

    training_labels = training_data["Win"]
    training_data = training_data.drop(["Win"], axis=1)
    testing_labels = testing_data["Win"]
    testing_data = testing_data.drop(["Win"], axis=1)
    print(f"\nTraining Labels:\n{training_labels}")
    print(f"\nTesting Labels:\n{testing_labels}")

    my_lr = LogisticRegression()
    my_lr_model = my_lr.fit(training_data, training_labels)
    my_model_predictions = my_lr_model.predict(testing_data)
    print(f"\nPredictions:\n{my_model_predictions}")
    print(f"\nActual Labels:\n{testing_labels}")

    my_cm = confusion_matrix(testing_labels, my_model_predictions)
    print(f"\nConfusion Matrix:\n{my_cm}")
    print(f"\nOriginal Probabilities:\n{my_lr_model.predict_proba(testing_data)}")

    return my_lr_model


def predict_vector(my_lr_model):
    """
    Predict the label category for a newly entered data vector
    :param my_lr_model: The linear regression model
    :return:
    """
    new_vector = []

    print()
    for curr_index in range(6):
        new_value = input(f"Enter value #{curr_index}: ")
        new_vector.append(int(new_value))

    column_names = ['Eliminations', 'Assists', 'Deaths', 'Damage', 'Healing', 'Mitigated']
    df = pd.DataFrame([new_vector], columns=column_names)

    predicted_label = my_lr_model.predict(df)
    print(f"\nPredicted Label: {predicted_label}")


def intro():
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
        elif choice.lower() == 't':
            model = train_data()
        elif choice.lower() == 'p':
            if model is None:
                print("Model does not exist. Make sure to train the model first!")
                continue
            predict_vector(model)
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid input.")


def main():
    intro()


if __name__ == '__main__':
    main()
