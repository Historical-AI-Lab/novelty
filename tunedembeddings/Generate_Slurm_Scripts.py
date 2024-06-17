# Generate slurm scripts

# reads in a slurm script and changes some parameters
# to run the same process on new data

with open('fin1990s.slurm', 'r') as file:
    data = file.readlines()

prefix = 'para20'
datasource = 'para20768'
resultfolder = 'para20_results'

decadefiles = dict()

for decade in range(1920, 2000, 10):
    decadefiles[decade] = []
    for line in data:
        if 'fin1990' in line:
            newline = line.replace('fin1990', prefix + str(decade))
            decadefiles[decade].append(newline)
        elif 'final768' in line:
            newline = line.replace('final768', datasource)
            newline = newline.replace('finalresults', resultfolder)
            start = str(decade)
            newline = newline.replace('1990', start)
            end = str(decade + 10)
            if end == '2000':
                end = '1997'
            newline = newline.replace('1997', end)
            decadefiles[decade].append(newline)
        else:
            decadefiles[decade].append(line)

for key, value in decadefiles.items():
    with open(prefix + str(key) + '.slurm', 'w') as file:
        file.writelines(value)
