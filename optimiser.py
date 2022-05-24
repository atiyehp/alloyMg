from copy import deepcopy
import numpy as np
from scipy.stats import truncnorm
import pickle
if 'google.colab' in str(get_ipython()):
    from alloyMg.model_paths import models
else:
    from model_paths import models



class AlDatapoint:
    def __init__(self, settings):
        self.categorical_inputs = settings.categorical_inputs
        self.categorical_inputs_info = settings.categorical_inputs_info
        self.range_based_inputs = settings.range_based_inputs

    def formatForInput(self):
        ht = [1 if i+1 in [*self.categorical_inputs.values()] else 0 for i in range(6)]
        my_input = [*self.range_based_inputs.values()] + ht
        return np.reshape(my_input, (1, -1))

    def print(self):
        for key, value in self.categorical_inputs.items():
            print(f"{key}: {self.categorical_inputs_info[key]['tag'][self.categorical_inputs_info[key]['span'].index(value)]}")
        for key, value in self.range_based_inputs.items():
            if value:
                print(f"{key}: {value}")

    def getAl(self):
        return 100 - sum(self.range_based_inputs.values())


class scanSettings:
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'DoS':
            self.loss_type = 'Linear'
            self.max_steps = 1
            self.targets = {
                'DoS': 10
            }
            self.categorical_inputs = {
                'heat treatment': [1]
            }
            self.categorical_inputs_info = {
                'heat treatment': {'span': [1, 2, 3, 4, 5, 6], 'tag': ['Extruded', 'ECAP',
                                                                       'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought']}}
            
            self.range_based_inputs = dict.fromkeys(
                ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
       'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
       'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'], [0])


        if self.mode == 'Mechanical':
            self.loss_type = 'Percentage'
            self.max_steps = 1
            self.targets = {
                'elongation%': 6,
                'tensile strength(MPa)': 250
            }
            self.categorical_inputs = {
                'heat treatment': [1, 2, 4, 5]
            }
            self.categorical_inputs_info = {
                'heat treatment': {'span': [1, 2, 3, 4, 5, 6], 'tag': ['Extruded', 'ECAP', 'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought']}}
            self.range_based_inputs = dict.fromkeys(
                ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
                 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
                 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'], [0])


class optimiser:
    def __init__(self, settings):
        self.step_batch_size = 100
        self.step_final_std = 0.01
        self.finetune_max_rounds = 3
        self.finetune_batch_size = 10
        self.mode = settings.mode
        self.loss_type = settings.loss_type
        self.targets = settings.targets
        self.max_steps = settings.max_steps
        self.categorical_inputs = settings.categorical_inputs
        self.range_based_inputs = settings.range_based_inputs
        self.settings = settings
        self.models = models

        self.run()

    def calculateLoss(self, datapoint):
        if self.mode == 'DoS':
            return self.models['elongation'].predict(datapoint.formatForInput())[0]
        elif self.mode == 'Mechanical':
            return self.models['elongation'].predict(datapoint.formatForInput())[0]

    def printResults(self, best_datapoint):
        if self.mode == 'DoS':
            print('data point:',best_datapoint.formatForInput()) 
            print('Results in a predicted %f elongation' % (self.models['elongation'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f yield' % (self.models['yield'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f tensile' % (self.models['tensile'].predict(best_datapoint.formatForInput())[0]))
        elif self.mode == 'Mechanical':
            print('Results in a predicted %f elongation' % (self.models['elongation'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f yield' % (self.models['yield'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f tensile' % (self.models['tensile'].predict(best_datapoint.formatForInput())[0]))

    def run(self):
        best_loss = None
        best_datapoint = AlDatapoint(self.settings)
        for i in range(self.max_steps):
            loss, datapoint = self.calculateStep(best_datapoint, i, 'all')
            if best_loss is None or loss < best_loss:
                best_datapoint = datapoint
                best_loss = loss

        for i in range(self.finetune_max_rounds):
            for key in [*self.categorical_inputs.keys(), *self.range_based_inputs.keys()]:
                loss, datapoint = self.calculateStep(best_datapoint, i, key)
                if loss < best_loss:
                    best_datapoint = datapoint
                    best_loss = loss
            else:
                break
        print('==========Scan Finished==========')
        self.printResults(best_datapoint)

    def calculateStep(self, best_datapoint, step_number, target_var):
        if target_var == 'all':
            batch_size = self.step_batch_size
        else:
            batch_size = self.finetune_batch_size
        loss = [0] * batch_size
        datapoints = []
        std = self.step_final_std * (self.max_steps / float(step_number + 1))
        for i in range(batch_size):
            datapoints.append(deepcopy(best_datapoint))
            for key in self.categorical_inputs.keys():
                if target_var == key or target_var == 'all':
                    datapoints[i].categorical_inputs[key] = np.random.choice(self.categorical_inputs[key])
            for key in self.range_based_inputs.keys():
                if target_var == key or target_var == 'all':
                    if max(self.range_based_inputs[key]) != min(self.range_based_inputs[key]):
                        a = (min(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        b = (max(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        datapoints[i].range_based_inputs[key] = round(
                            float(truncnorm.rvs(a, b, loc=np.mean(best_datapoint.range_based_inputs[key]), scale=std)),
                            2)
                    else:
                        datapoints[i].range_based_inputs[key] = min(self.range_based_inputs[key])
            loss[i] = self.calculateLoss(datapoints[i])
        return min(loss), datapoints[loss.index(min(loss))]
