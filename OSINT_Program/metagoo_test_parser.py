import re

# consider doing something with re i.e. grabbing the header 'users' or 'software' directly from the sentance (using reg experessions) for user friendliness

file = open('meta_output_test', 'r')

listOfLines = file.read().split('\n')

indi_list = []

task_list = []

final = {}

#test = re.search(r'(?<=List of (.*) found:).*', listOfLines)


"""
for x in listOfLines:
    if '[+]' in x:
        indi_list.append(listOfLines.index(x))
        result = re.search('List of (.*) found:', x)
        task_list.append(result.group(1))

print(indi_list)
print(task_list)




for n in indi_list:
    if indi_list.index(n) < len(indi_list) - 1:
        print(listOfLines[n + 2: indi_list[indi_list.index(n) + 1]])

    else:
        print(listOfLines[n + 2:])

"""