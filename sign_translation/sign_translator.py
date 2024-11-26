import csv
import os

my_path = os.path.abspath(os.path.dirname(__file__))

class SignTranslator:
    def __init__(self):
        self.signs = self.parseTXTFile("models/labels.txt")

    def parseTXTFile(self, filename):
        '''
        Parses a TXT file located in te config directory and returns its contents as a dictionary.

        @param filename: The name of the csv file to parse.

        @return the contents of the parsed file.
        '''
        list_labels = []
        with open(filename, 'r') as f:
            for line in f:
                list_labels.append(line)
        return list_labels
    
    def get_sign(self, s):
        '''
        Returns the sign name for a given sign id.

        @param s: the sign id

        @return the sign name
        '''
        return self.signs[s]