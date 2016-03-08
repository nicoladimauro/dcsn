# adapted from scikit-multilearn for Python3 and meka1.9
# https://github.com/scikit-multilearn/

import subprocess
import numpy as np
import math

class Meka(object):
    """ Runs the MEKA classifier
        Parameters
        ----------
        meka_classifier : string
            The MEKA classifier string and parameters from the MEKA API, such as: "meka.classifiers.multilabel.MULAN -S RAkEL2"
        
        weka_classifier : string
            The WEKA classifier string and parameters from the WEKA API, such as: "weka.classifiers.trees.J48"
        
        java_command : string
            Path to test the java command
        meka_classpath: string
            Path to the MEKA class path folder, usually the folder lib in the directory MEKA was extracted to
    """
    def __init__(self, meka_classifier = None, weka_classifier = None, java_command = '/usr/bin/java', meka_classpath = "/home/niedakh/icml/meka-1.7/lib/", threshold = 0.5):
        super(Meka, self).__init__()

        self.java_command = java_command
        self.classpath = meka_classpath
        self.meka_classifier = meka_classifier
        self.threshold = threshold
        self.verbosity = 20
        self.weka_classifier = weka_classifier
        self.output = None
        self.warnings = None
        self.results = None
        self.statistics = None

    def run(self, train_file, test_file):
        """ Runs the meka classifiers
        Parameters
        ----------
        train_file : string
            Path to train .arff file in meka format (big endian, labels first in attributes list).
        
        test_file : string
            Path to test .arff file in meka format (big endian, labels first in attributes list).
        Returns
        -------
        predictions: array-like of array-likes of {0,1}
            array of binary label vectors including label predictions
        extra_output: dictionary
            dictionary of additional output generated by meka, consult meka documentation for more information
        """
        self.output = None
        self.warnings = None

        # meka_command_string = 'java -cp "/home/niedakh/pwr/old/meka-1.5/lib/*" meka.classifiers.multilabel.MULAN -S RAkEL2  
        # -threshold 0 -t {train} -T {test} -verbosity {verbosity} -W weka.classifiers.bayes.NaiveBayes'
        # meka.classifiers.multilabel.LC, weka.classifiers.bayes.NaiveBayes
        meka_command_string = '{java} -cp "{classpath}*" {meka} -threshold PCut1 -t {train} -T {test} -verbosity {verbosity} -W {weka}'
        

        input_files = {
            'java': self.java_command,
            'meka': self.meka_classifier,
            'weka': self.weka_classifier,
            'train': train_file,
            'test': test_file,
            'threshold': self.threshold,
            'verbosity': self.verbosity,
            'classpath': self.classpath
        }
        meka_command = meka_command_string.format(**input_files)
        print(meka_command)
        pipes = subprocess.Popen(meka_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = pipes.communicate()


#        print(output, error)
        if pipes.returncode != 0:
            raise Exception
        
        self.output = output
        self.parse_output()
        return self.results, self.statistics

    def parse_output(self):
        """ Internal function for parsing MEKA output."""
        if self.output is None:
            self.results = None
            self.statistics = None
            return None

        predictions_split_head = "==>\n"
        predictions_split_foot = "|==============================<\n"


        self.warnings = self.output.decode('utf-8').split(predictions_split_head)[0]

        # predictions, first split
        predictions = self.output.decode('utf-8').split(predictions_split_head)[1].split(predictions_split_foot)[0]
        # then clean up and remove empty lines
        predictions = filter(lambda x: len(x), predictions.replace('\n\n','\n').split('\n'))
        # parse into list of row classifications

        self.results = np.array(
            [item.split('[ ')[2].split(' ]')[0].replace('0,','0.').split() for item in predictions]).astype(float)

        # split, cleanup, remove empty lines
        statistics = self.output.decode('utf-8').split(predictions_split_head)[1].split(predictions_split_foot)[1]
        statistics = filter(lambda x: len(x), statistics.replace('\r','\n').replace('\n\n','\n').split('\n'))
        statistics = [item[:31]+":"+item[31:] for item in statistics if '==' not in item]
        statistics = [item.replace(' ', '') for item in statistics]
        
        self.statistics = {}

        for item in statistics:
            s = item.split(":")
            if len(s)>1:
                self.statistics[s[0]]=s[1]
        print("Threshold: ", float(self.statistics['Threshold']))
        self.threshold = float(self.statistics['Threshold'])
        self.results = (self.results >= self.threshold).astype(int)







