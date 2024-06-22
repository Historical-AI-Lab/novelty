# Generate slurm scripts

# reads in a slurm script and changes some parameters
# to run the same process on new data

with open('fic1950-54.slurm', 'r') as file:
    data = file.readlines()

decadefiles = dict()

for decade in range(1910, 1985, 5):
    # print(decade)
    newspan = str(decade) + '-' + str(decade + 4)[-2:]
    if newspan == '1950-54':
        continue
    print(decade, newspan)
    decadefiles[decade] = []
    for line in data:
        if '1950-54' in line:
            newline = line.replace('1950-54', newspan)
            decadefiles[decade].append(newline)
        elif 'fiction/doctopics' in line:
            newline = newline.replace('-e 1955', "-e " + str(decade + 5))
            newline = line.replace('-s 1950', '-s ' + str(decade))
            decadefiles[decade].append(newline)
        else:
            decadefiles[decade].append(line)

for decade, value in decadefiles.items():
    if decade == 1950:
        continue
    print(decade, len(value))
    newspan = str(decade) + '-' + str(decade + 4)[-2:]
    with open('fic' + newspan + '.slurm', 'w') as file:
        file.writelines(value)
