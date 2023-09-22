import re

# consider doing something with re i.e. grabbing the header 'users' or 'software' directly from the sentance (using reg experessions) for user friendliness

file = open('meta_output_test', 'r')

listOfLines = file.read().split('[+]')

for i in range(1, len(listOfLines)):
    print(listOfLines[i])
