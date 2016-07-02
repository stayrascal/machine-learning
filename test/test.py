def filterDataOf678Months(originFile='rain.txt', filterFile='filter.txt'):
    fr = open(originFile)
    outfile = open(filterFile, 'a')
    index = 0
    for line in fr.readlines():
        listFromLine = textParse(line)
        if int(listFromLine[0]) in [6, 7, 8]:
            outfile.write(line)
        index += 1
    outfile.close()


def textParse(data):
    import re
    listOfToken = re.split(r'\D', data)
    return [token for token in listOfToken if len(token) > 0]


def sumThridColum(filename):
    fr = open(filename)
    years = {}
    sum = 0
    for line in fr.readlines():
        listFromLine = textParse(line)
        if listFromLine[2] not in years:
            years[listFromLine[2]] = int(listFromLine[3])
        else:
            years[listFromLine[2]] += int(listFromLine[3])
        sum += int(listFromLine[3])
    years_names = sorted(years.keys())
    sum_per_years = []
    avg_per_years = []
    avg_per_years.append('year\tavg_distance')
    avg = float(sum / 50)
    for year in years_names:
        sum_per_years.append(year + "\t" + str(years[year]))
        avg_per_years.append(
            year + "\t" + str(format(float((years[year] - avg) / avg), '.2f')))

    return sum_per_years, avg_per_years


def test():
    sum_per_years, avg_per_years = sumThridColum("test.txt")
    fw = open("sum_per_years.txt", "w")
    favg = open("avg_per_years.txt", "w")
    fw.write('\n'.join(sum_per_years))
    favg.write('\n'.join(avg_per_years))
    fw.close()


def main():
    filterDataOf678Months(originFile='S201604302106062744600.txt')


def test1():
    import pandas as pd
    data = pd.read_fwf('test.txt')
    print(data.describe())
