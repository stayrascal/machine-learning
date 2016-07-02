#!/usr/bin/python

import pandas as pd


def getData(filename):
    return pd.read_csv(filename)


def showHead(data):
    return data.head()


def showTotalSalariesPerTeamPerYear(data):
    return data.groupby(['teamID', 'yearID']).sum()


def mergeSalariesAndTeams(salaries, teams):
    return pd.merge(teams, salaries, how="left", on=['teamID', 'yearID'])


def main():
    salaries = getData('Salaries.csv')
    teams = getData('Teams.csv')
    # print(showHead(salaries))
    # print(showHead(teams))
    total_salaries = showTotalSalariesPerTeamPerYear(salaries)
    # print(total_salaries)
    # print(showHead(total_salaries))
    merege = mergeSalariesAndTeams(salaries, teams)
    print(merege[['W', 'salary']])


if __name__ == '__main__':
    main()
