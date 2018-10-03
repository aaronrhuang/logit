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
    sql = "SELECT repoUrl, repository, a.json as ajson, b.json as bjson, a.id\
    FROM (SELECT * FROM GetRepository LIMIT 5000) a INNER JOIN (SELECT * FROM GetCommitStats LIMIT 5000) b\
    ON b.repoUrl LIKE CONCAT('%', a.repository)\
    WHERE b.json IS NOT NULL AND a.json IS NOT NULL\
    LIMIT 10000"

    cursor.execute(sql)
    fetched = cursor.fetchmany(10000)
    print len(fetched)
    statDict = {}
    for r in fetched:
        da = json.loads(r['ajson'])
        db = json.loads(r['bjson'])[10] if r and r['bjson'] else {}
        astats = {key:da[key] for key in GIT_REPO_FIELDS if key in da}
        bstats = {key:db[key] for key in GIT_STAT_FIELDS if key in db}
        astats.update(bstats)
        statDict[r['repoUrl']] = astats

    print len(statDict)

    return statDict

# Create the training data/labels with respect to some field to predict
def get_variables(y_var, statDict):
    X, Y, Xv, Yv = [],[],[],[]
    for repo, fields in statDict.items()[:-100]:
        if y_var in fields:
            row = []
            for i in GIT_STAT_FIELDS:
                if i in fields:
                    row += [fields[i]]
                else:
                    row += [0]
            X.append(row)
            Y.append(min(100,fields[y_var]))
    for repo, fields in statDict.items()[-100:]:
        if y_var in fields:
            row = []
            for i in GIT_STAT_FIELDS:
                if i in fields:
                    row += [fields[i]]
                else:
                    row += [0]
            Xv.append(row)
            Yv.append(min(100,fields[y_var])) if y_var in fields else Yv.append(0)
    return (np.array(X),np.array(Y), np.array(Xv), np.array(Yv))

if __name__ == "__main__":
    try:
        with connection.cursor() as cursor:

            statDict = get_dicts(cursor)

            X,Y,Xv,Yv = get_variables('stargazers_count', statDict)
            print X.shape,Y.shape, Xv.shape, Yv.shape
            reg = linear_model.LogisticRegression(penalty='l2', C=10, intercept_scaling=1)
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