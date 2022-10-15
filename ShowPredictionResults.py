import shutil

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


class ShowPrediction:
    def __init__(self):
        # get result file(.npy file)
        all_predictions = [pred for pred in os.listdir('predictions/')]
        current_prediction_directory = max(all_predictions)
        result_file = ''
        for file in os.listdir('predictions/{}/'.format(current_prediction_directory)):
            if '.npy' in file:
                result_file = 'predictions/{0}/{1}'.format(current_prediction_directory, file)
                break

        # create output directory
        # if os.path.exists('results/{}'.format(current_prediction_directory)):
        #     shutil.rmtree('results/{}'.format(current_prediction_directory))
        # os.mkdir('results/{}'.format(current_prediction_directory))


        # load .npy file
        data_array = np.load(result_file)
        print(data_array[0])

        # save the result to data frame to plot it in bar chart
        ser = pd.Series(data_array[0])  # index=0 because the data_array contains result for one test image only
        data = pd.DataFrame()
        data = data.append(ser, ignore_index=True)
        data.columns = ['Pleural-Thickening', 'Edema', 'Consolidation', 'Atelectasis', 'Effusion', 'Normal']
        data.to_csv('data.csv', index=False)

        # generate a bar chart of the result
        percentages = ser.values
        plt.figure(figsize=(8, 8))
        colors_list = ['Red', 'Orange', 'Blue', 'Purple', 'Green', 'Yellow']
        graph = plt.bar(data.columns, percentages, color=colors_list)

        i = 0
        for p in graph:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(x + width / 2, y + height * 1.01, str(format(percentages[i] * 100, ".2f")) + '%', ha='center')
            i += 1
        # plt.show()
        plt.savefig('result.png')
