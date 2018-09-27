import numpy as np
from sklearn import linear_model

import json
import pymysql.cursors
from django.core.cache import cache

import matplotlib.pyplot as plt
from credentials import *


# Connect to the database
connection = pymysql.connect(host=DATACORE_HOST,
                             user=USER,
                             password=PASSWORD,
                             db=DB,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

GIT_STAT_FIELDS = ['number_of_files', 'number_of_lines', 'number_of_comments', 'number_of_tests', 'number_of_bimaries', 'dev_ops_score', 'readability_score']
GIT_REPO_FIELDS = ['forks_count', 'stargazers_count', 'subscribers_count']

# parse database rows/json into dictionaries,
# TODO: add caching
def get_dicts(cursor):
    cursor.execute("SELECT `*` FROM `GetCommitStats` LIMIT 1000")
    resultStat = cursor.fetchmany(1000)
    statDict = {}
    for r in resultStat:
        d = json.loads(r['json'])[10]
        statDict[r['repoUrl'].split('/')[-1]] = {key:d[key] for key in GIT_STAT_FIELDS if key in d}

    cursor.execute("SELECT `*` FROM `GetRepository` LIMIT 1000")
    resultRepo = cursor.fetchmany(1000)
    repoDict = {}
    for r in resultRepo:
        d = json.loads(r['json'])
        repoDict[r['repository']] = {key:d[key] for key in GIT_REPO_FIELDS if key in d}

    return (statDict,repoDict)

# Create the training data/labels with respect to some field to predict
def get_variables(y_var, statDict, repoDict):
    X, Y, Xv, Yv = [],[],[],[]
    for repo, fields in statDict.items()[:-100]:
        if repo in repoDict:
            row = []
            for i in GIT_STAT_FIELDS:
                if i in fields:
                    row += [fields[i]]
                else:
                    row += [0]
            X.append(row)
            Y.append(repoDict[repo][y_var]) if y_var in repoDict[repo] else Y.append(0)
    for repo, fields in statDict.items()[-100:]:
        if repo in repoDict:
            row = []
            for i in GIT_STAT_FIELDS:
                if i in fields:
                    row += [fields[i]]
                else:
                    row += [0]
            Xv.append(row)
            Yv.append(repoDict[repo][y_var]) if y_var in repoDict[repo] else Yv.append(0)
    print(np.array(X).shape,np.array(Y).shape, np.array(Xv).shape, np.array(Yv).shape)
    return (np.array(X),np.array(Y), np.array(Xv), np.array(Yv))

if __name__ == "__main__":
    try:
        with connection.cursor() as cursor:

            statDict,repoDict = get_dicts(cursor)

            X,Y,Xv,Yv = get_variables('stargazers_count', statDict, repoDict)

            reg = linear_model.LogisticRegression(penalty='l2', C=5, intercept_scaling=1)
            reg.fit(X, np.ravel(Y))
            w,w_0 = reg.coef_, reg.intercept_

            plt.plot([i for i in range(len(Y))],Y, alpha=0.7, label = "Actual")
            plt.plot([i for i in range(len(Y))],reg.predict(X), alpha = 0.7, label = "Predicted")
            plt.xlabel('Repository')
            plt.ylabel('Stargazers Count')
            plt.legend()
            plt.savefig("figs/logr_stargazers_train.png")
            plt.show()

            plt.plot([i for i in range(len(Yv))],Yv, alpha=0.7, label = "Actual")
            plt.plot([i for i in range(len(Yv))],reg.predict(Xv), alpha = 0.7, label = "Predicted")
            plt.xlabel('Repository')
            plt.ylabel('Stargazers Count')
            plt.legend()
            plt.savefig("figs/logr_stargazers_val.png")
            plt.show() 

    finally:
        connection.close()