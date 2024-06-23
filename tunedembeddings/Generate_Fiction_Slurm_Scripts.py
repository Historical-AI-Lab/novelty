# Generate slurm scripts

# reads in a slurm script and changes some parameters
# to run the same process on new data

with open('fic1910-13.slurm', 'r') as file:
    data = file.readlines()

prefix = 'fic'

decadefiles = dict()

for floor in range(1913, 2000, 3):
    decadefiles[floor] = []
    newspan = str(floor) + '-' + str(floor + 3)[-2:]
    for line in data:
        if '1910-13' in line:
            newline = line.replace('1910-13', newspan)
            decadefiles[floor].append(newline)
        elif 'fic768' in line:
            newline = line.replace('1910', str(floor))
            newline = line.replace('1913', str(floor + 3))
            decadefiles[floor].append(newline)
        else:
            decadefiles[floor].append(line)

for key, value in decadefiles.items():
    with open('fic' + str(key) + '-' + str(key + 3) + '.slurm', 'w') as file:
        file.writelines(value)
