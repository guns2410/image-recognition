from os import system

import pandas

from image_processor import predict_types, detect_objects

_ = system('clear')

print("--------------------------------")
print('Ready to Predict and Detect')

process_images = True


def print_prediction(prediction_image_path):
    print('\n> Predicting Types\n')
    predictions = predict_types(prediction_image_path)

    df = pandas.DataFrame(data=predictions, columns=['Predictions', 'Probabilities'])
    df.set_index('Predictions', inplace=True)
    df = df.sort_values('Probabilities', ascending=False)

    print(df)


def print_detection(detection_image_path):
    print('\n> Detecting Objects\n')
    detections = detect_objects(detection_image_path)

    df = pandas.DataFrame(detections)
    df.set_index('name', inplace=True)
    df = df.drop(['box_points'], axis=1)
    df = df.sort_values('percentage_probability', ascending=False)

    print(df)


while process_images:
    image_path = input('\n\nInput the image you want to analyze: ')
    action = input(
        '\nEnter 1 to Predict image type\nEnter 2 to detect objects from image\nEnter 3 to do both, Predict and Detect\n\nWhat would you like to do?: '
    )

    # Check the condition as to what action needs to be performed
    if action == '1':
        print_prediction(image_path)
    elif action == '2':
        print_detection(image_path)
    elif action == '3':
        print_prediction(image_path)
        print_detection(image_path)

    # See if another iteration is required
    proceed = input('\n\nAnalyze another image? (y/n): ')
    if proceed == 'n':
        process_images = False

    _ = system('clear')
