import networkx as nx
import json
import sys
import time
import ast
import pickle
import numpy as np
from datetime import datetime
import math
from sklearn.cluster import KMeans
import operator
import pandas as pd
import re
from collections import Counter
import argparse


class Groups:
    def __init__(self, users, prods):
        self.users = users
        self.prods = prods

    def __lt__(self, other):
        return len(self.users) < len(other.users)


class Review:
    def __init__(self, userid, useridmapped, prodid, prodidmapped, rating, label, date, content):
        self.userid = userid
        self.useridmapped = useridmapped
        self.prodid = prodid
        self.prodidmapped = prodidmapped
        self.rating = rating
        self.label = label
        self.date = date
        self.content = content

    def __repr__(self):
        return '({})'.format(self.prodid)

    def __hash__(self):
        return hash(self.prodid)

    def __eq__(self, other):
        return self.prodid == other.prodid

    def __ne__(self, other):
        return not self.__eq__(other)


text = []
filee = open("../data/reviewContent_yelpnyc", 'r', encoding="utf-8")
for f in filee:
    fsplit = f.split("\t")
    text.append(fsplit[3].strip())

filee.close()

allprods = {}
allusers = {}
reviewtime = {}
reviewrating = {}
reviewcontent = {}
wholerev = {}
minn = {}
d = {}
fake = set()
rvdate = {}
maxrvdate = {}
maxrvcon = {}
c = 0
filee = open("../data/metadata_yelpnyc", 'r')
for f in filee:

    fsplit = f.split("\t")

    userid = int(fsplit[0])
    prodid = int(fsplit[1])
    rating = int(round(float(fsplit[2])))
    label = fsplit[3]

    if int(label) == -1:
        fake.add(userid)

    date = fsplit[4].strip()
    date = datetime.strptime(date, "%Y-%m-%d").date()
    if prodid not in d:
        minn[prodid] = 0
        d[prodid] = date

    minn[prodid] = date
    if minn[prodid] < d[prodid]:
        d[prodid] = minn[prodid]

    if userid not in rvdate:
        rvdate[userid] = {}
        maxrvdate[userid] = {}
        maxrvcon[userid] = {}
    if prodid not in rvdate[userid]:
        rvdate[userid][prodid] = date
        maxrvdate[userid][prodid] = date
        maxrvcon[userid][prodid] = text[c]

    rvdate[userid][prodid] = date
    if rvdate[userid][prodid] > maxrvdate[userid][prodid]:
        maxrvdate[userid][prodid] = rvdate[userid][prodid]
        maxrvcon[userid][prodid] = text[c]
    c = c + 1

filee.close()

c = 0
filee = open("../data/metadata_yelpnyc", 'r')
for f in filee:

    fsplit = f.split("\t")

    userid = int(fsplit[0])
    prodid = int(fsplit[1])
    rating = int(round(float(fsplit[2])))
    label = fsplit[3]
    date = fsplit[4].strip()

    newdate = datetime.strptime(date, "%Y-%m-%d").date()
    if newdate == maxrvdate[userid][prodid]:

        datetodays = (newdate - d[prodid]).days

        r = Review(userid, '', prodid, '', rating, label, datetodays, maxrvcon[userid][prodid])

        if userid not in reviewtime:
            reviewtime[userid] = {}
        if prodid not in reviewtime[userid]:
            reviewtime[userid][prodid] = datetodays
        if userid not in reviewrating:
            reviewrating[userid] = {}
        if prodid not in reviewrating[userid]:
            reviewrating[userid][prodid] = rating
        if userid not in reviewcontent:
            reviewcontent[userid] = {}
        if prodid not in reviewcontent[userid]:
            reviewcontent[userid][prodid] = maxrvcon[userid][prodid]
        if userid not in allusers:
            allusers[userid] = []
        if prodid not in allprods:
            allprods[prodid] = []
        if userid not in wholerev:
            wholerev[userid] = {}
        if prodid not in wholerev[userid]:
            wholerev[userid][prodid] = r

        allprods[prodid].append(userid)
        allusers[userid].append(prodid)

        c = c + 1
filee.close()


def reviewtightness(group, L):
    v = 0
    for user in group.users:
        for prod in group.prods:
            # prod=prod.split("_")[0]
            if prod in reviewtime[user]:
                v = v + 1
    if len(list(group.prods)) == 0:
        return 0
    return (v * L) / (1.0 * len(list(group.users)) * len(list(group.prods)))


def neighbortightness(group, L):
    userlist = list(group.users)
    denom = 0
    num = 0
    for user1i in range(len(userlist)):
        user1 = userlist[user1i]
        for user2i in range(user1i + 1, len(userlist)):
            user2 = userlist[user2i]
            union = set(allusers[user1]).union(set(allusers[user2]))
            intersection = set(allusers[user1]).intersection(set(allusers[user2]))
            num = num + len(intersection) / (len(union) * 1.0)
            denom = denom + 1

    return (num * L) / (1.0 * denom)


def producttightness(group):
    c = 0
    userlist = list(group.users)
    for user in userlist:
        if c == 0:
            intersection = set(allusers[user])
            union = set(allusers[user])
        else:
            intersection = intersection.intersection(set(allusers[user]))
            union = union.union(set(allusers[user]))
        c = c + 1

    return len(intersection) / (len(union) * 1.0)


#
def averagetimewindow_ratingvariance(group, L):
    if len(list(group.prods)) == 0:
        return 0, 0
    avg = 0
    var = 0
    for prod in group.prods:
        prodlist = []
        prodtym = []
        # prod=prod.split("_")[0]
        minn = float('inf')
        maxx = 0
        for user in group.users:
            if prod in reviewtime[user]:
                prodlist.append(reviewrating[user][prod])
                prodtym.append(reviewtime[user][prod])

        var = var + np.var(prodlist)
        ans = np.std(prodtym)
        if ans < 30:
            avg = avg + (1 - ans / 30.0)

    var = var / (-1.0 * len(group.prods))
    rating_variance = 2 * (1 - (1.0 / (1 + math.exp(var))))

    return (avg * L) / (1.0 * len(list(group.prods))), rating_variance * L


def productreviewerratio(group):
    maxx = 0

    for prod in group.prods:

        num = 0
        denom = 0
        for user in group.users:
            if prod in reviewtime[user]:
                num = num + 1  #

        for r in allprods[prod]:
            # if int(r.rating)==int(prod.split("_")[1]):
            denom = denom + 1

        ans = num / (1.0 * denom)
        if ans > maxx:
            maxx = ans

    return maxx


def groupsize(group):
    return 1 / (1 + math.exp(3 - len(list(group.users))))


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)


def cosine(content_a, content_b):
    text1 = content_a
    text2 = content_b

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine_result = get_cosine(vector1, vector2)
    return cosine_result


def GCS(group):
    maxx = 0
    for prod in group.prods:
        avg = 0
        c = 0
        userlist = list(group.users)
        for r1i in range(len(userlist)):
            r1 = userlist[r1i]
            if prod in reviewtime[r1]:
                for r2i in range(r1i + 1, len(userlist)):
                    r2 = userlist[r2i]
                    if prod in reviewtime[r2]:
                        avg = avg + cosine(reviewcontent[r1][prod], reviewcontent[r2][prod])
                        c = c + 1
        if c != 0:
            avg = avg / (c * 1.0)
        if avg > maxx:
            maxx = avg

    return maxx


#
def GMCS(group):
    avg = 0
    totc = len(list(group.users))

    if len(group.prods) <= 1:
        return 0
    for user in group.users:
        ans = 0
        c = 0
        prodlist = list(group.prods)
        for p1i in range(len(prodlist)):
            p1 = prodlist[p1i]
            if p1 in reviewtime[user]:
                for p2i in range(p1i + 1, len(prodlist)):
                    p2 = prodlist[p2i]

                    if p2 in reviewtime[user]:
                        ans = ans + cosine(reviewcontent[user][p1], reviewcontent[user][p2])
                        c = c + 1
        if c != 0:
            avg = avg + (ans * 1.0) / c
        else:
            totc = totc - 1
    if totc == 0:
        return 0
    return (avg * 1.0) / (totc)


def calc_score(g, Lsub):
    #score = []
    ans = averagetimewindow_ratingvariance(g, Lsub)
    score = [reviewtightness(g, Lsub), neighbortightness(g, Lsub), producttightness(g), ans[0], ans[1],
             productreviewerratio(g)]
    return score, sum(score) / (len(score) * 1.0)


def create_groups():
    ass = np.load("assignments.npy")
    clusters = ass.argmax(axis=1)
    x = 0
    v = 5
# grps[len(grps)]={'id':len(grps),'users':list(userset),'prods':list(prodset),'scorepred':0, 'scoregt':0, 'fakegt':0,'fakepred':0}
    fgrps = {}
    for i in range(100):
        fgrps[i]={}
        fgrps[i]['id'] = i+1#count
        fgrps[i]['users'] = set()
        fgrps[i]['prod'] = set()
        fgrps[i]['scorepred']=0
        fgrps[i]['scoregt']=0
        fgrps[i]['fakegt']=0

    with open('dictprod.pickle', 'rb') as handle:
        dictprod = pickle.load(handle)

    with open('prodlist.pickle', 'rb') as handle:
        prodlist = pickle.load(handle)

    for i in range(100):#存每组的prd
        for j in range(clusters.shape[0]):
            if clusters[j][i] == 1:#i--组号 j--prod_rating号
                prod_str = dictprod[j+1]
                strlist = prod_str.split('_')
                fgrps[i]['prod'].add(strlist[0])
                fgrps[i]['users'].add(tuple(prodlist[prod_str]))

    for grp in fgrps:

        if len(list(fgrps[grp]['users'])) > 1:

            group = Groups(fgrps[grp]['users'], fgrps[grp]['prods'])

            Lsub = 1.0 / (1 + (math.exp(3 - len(list(group.users)) - len(list(group.prods)))))#punction
            ans = calc_score(group, Lsub)
            scorepred = ans[0]
            spamicity = ans[1]

            c = 0
            denom = 0
            for u in group.users:
                if u in fake:
                    c = c + 1
                denom = denom + 1
            store = (c * 1.0) / denom

            c = 0
            denom = 0
            for u in group.users:
                for p in group.prods:
                    if p in wholerev[u]:
                        if int(wholerev[u][p].label) == -1:
                            c = c + 1
                        denom = denom + 1
            if len(list(group.prods)) == 0:
                denom = 1
                c = 0
            if x not in grps:
                grps[x] = {'id': x, 'users': list(group.users), 'prods': list(group.prods), 'scorepred': scorepred,
                           'scoregt': store, 'scoregtreviewprec': (c * 1.0) / denom, 'fakegt': 0,
                           'fakepred': spamicity}
                x = x + 1

    tc = 0
    for grp in grps:
        scorepred = grps[grp]['scorepred']
        summ = sum(scorepred[:v]) / 6.0
        if summ > 0.4:
            tc = tc + 1

    if tc < 100:
        v = v + 1

    for grp in grps:
        scorepred = grps[grp]['scorepred']
        summ = sum(scorepred[:v]) / 6.0
        if summ > 0.4:
            grps2[len(grps2)] = grps[grp]
            grps2[len(grps2) - 1]['id'] = len(grps2) - 1


grps = {}
grps2 = {}
create_groups()
with open("./outputgroups", 'w') as fp:
    json.dump(grps2, fp)
print('end')
