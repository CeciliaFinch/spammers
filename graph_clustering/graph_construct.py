import networkx as nx
import json
import sys
import time
from datetime import datetime
import pickle
import operator
import numpy as np
import simplejson as json

CC_mapper = {}


class Review:
    def __init__(self, userid, useridmapped, prodid, prodidmapped, rating, label, date):
        self.userid = userid
        self.useridmapped = useridmapped
        self.prodid = prodid
        self.prodidmapped = prodidmapped
        self.rating = rating
        self.label = label
        self.date = date

    def __repr__(self):
        return '({},{})'.format(self.userid)

    def __hash__(self):
        return hash((self.userid))

    def __eq__(self, other):
        return self.userid == other.userid

    def __ne__(self, other):
        return not self.__eq__(other)


def load_graph():

    text = {}
    allusers = {}
    textf = open('../data/reviewContent_yelpnyc', 'r', encoding='UTF-8')
    for row in textf:

        userid = int(row.split("\t")[1].strip())
        prodid = int(row.split("\t")[0].strip())
        if userid not in text:
            text[userid] = {}
            allusers[userid] = []
        if prodid not in text[userid]:
            text[userid][prodid] = row.split("\t")[3].strip()
        if prodid not in allusers[userid]:
            allusers[userid].append(prodid)

    minn = {}
    d = {}
    fake = set()
    filee = open('../data/metadata_yelpnyc', 'r')
    for f in filee:

        fsplit = f.split("\t")

        userid = fsplit[0]
        prodid = fsplit[1]
        rating = int(round(float(fsplit[2])))
        label = fsplit[3]

        if int(label) == -1:
            fake.add(userid)

        date = fsplit[4].strip()
        if prodid not in d:
            minn[prodid] = 0
            d[prodid] = datetime.strptime(date, "%Y-%m-%d").date()

        minn[prodid] = datetime.strptime(date, "%Y-%m-%d").date()
        if minn[prodid] < d[prodid]:
            d[prodid] = minn[prodid]
    filee.close()

    G = nx.Graph()
    reviewsperproddata = {}
    nodedetails = {}
    prodlist = {}
    dictprod = {}
    dictprodr = {}
    mainnodelist = set()
    count = 0
    filee = open('../data/metadata_yelpnyc', 'r')
    for f in filee:

        fsplit = f.split("\t")

        userid = fsplit[0]
        prodid = fsplit[1]
        rating = str(int(round(float(fsplit[2]))))
        label = fsplit[3]
        date = fsplit[4].strip()
        newdate = datetime.strptime(date, "%Y-%m-%d").date()
        datetodays = (newdate - d[prodid]).days
        review = Review(userid, '', prodid, '', rating, label, datetodays)

        if prodid + "_" + rating not in reviewsperproddata:
            count = count + 1
            reviewsperproddata[prodid + "_" + rating] = set()
            dictprod[count] = prodid + "_" + rating
            dictprodr[prodid + "_" + rating] = count
            prodlist[prodid + "_" + rating] = []
            G.add_node(count)

        prodlist[prodid + "_" + rating].append(review)

    filee.close()
    feature = np.zeros((count, 161148), dtype=float, order='C')
    for i in range(count):
        detail_pr = dictprod[i+1] #prod_rating
        reviewList = prodlist[detail_pr]
        for review in reviewList:
            feature[count-1][int(review.userid)] = 1


    # count：prod_rating
    # G = nx.Graph() node: count
    # nodedetails={}
    # prodlist={} 键：prodid_rating 值：review
    # dictprod={} 键：count 值：prodid_rating 记录用
    # dictprodr={} 键：prodid_rating 值：count
    # mainnodelist=set()

    with open('../refine_groups/dictprod.pickle', 'wb') as handle:
        pickle.dump(dictprod, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../refine_groups/prodlist.pickle', 'wb') as handle:
        pickle.dump(prodlist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    edgedetails = {}
    cmnrevrsedges = {}
    cmnrevrslist = {}
    cmnrevrslistr = {}
    cmnrevrsedgeslen = {}
    countt = 0
    visited = {}
    mark = {}
    graphlist = list(G.nodes())
    cr = {}
    for node1i in range(len(graphlist)):
        node1 = graphlist[node1i]
        if node1 not in cr:
            cr[node1] = []
        for u1i in range(len(prodlist[dictprod[node1]])):
            u1 = prodlist[dictprod[node1]][u1i]
            cr11 = set()
            cr11.add(u1)
            for u2i in range(u1i + 1, len(prodlist[dictprod[node1]])):
                u2 = prodlist[dictprod[node1]][u2i]
                if abs(u1.date - u2.date) < 10:
                    cr11.add(u2)
            cr[node1].append(cr11)
        cr[node1].sort(key=len, reverse=True)
    # crll该评论及10天内对应的评论的集合
    # cr:键--nodel(prodid_rating) 值：列表-crll--10天内（abs）评论数量排序
    # nodel--->所有评论及对应的10天内的评论，按10天内评论数排序

    edgecount = {}
    for node1i in range(len(graphlist)):

        node1 = graphlist[node1i]
        for node2i in range(node1i + 1, len(graphlist)):
            node2 = graphlist[node2i]

            maxx = 0
            maxxcr = set()

            cr1 = cr[node1]
            cr2 = cr[node2]
            crlist = set()
            f = 0
            for cri1 in cr1:
                if len(cri1) < 2:
                    break
                for cri2 in cr2:
                    if len(cri2) < 2:
                        f = 1
                        break
                    crr = cri1.intersection(cri2)
                    crr = frozenset(crr)
                    if len(crr) > 1:
                        crlist.add(crr)

                if f == 1:
                    break

            crlist = list(crlist)  # crlist: 两个prod-rating的评论里的cmnrevwgrp
            crlist.sort(key=len, reverse=True)

            for commonreviewers in crlist:
                if len(commonreviewers) > 1:

                    if commonreviewers not in cmnrevrslistr:
                        countt = countt + 1
                        cmnrevrslist[countt] = commonreviewers
                        cmnrevrslistr[commonreviewers] = countt
                        maincount = countt  # countt不加一，maincount: grps数
                    else:
                        maincount = cmnrevrslistr[commonreviewers]
                    if node1 < node2:
                        n1 = node1
                        n2 = node2
                    else:
                        n1 = node2
                        n2 = node1
                    # n1<n2
                    if maincount not in cmnrevrsedges:
                        cmnrevrsedges[maincount] = []
                    # cmnrevrsedges: 键 maincount(grps的id-maincount) 值 (node1,node2)
                    # cmnrevrlist: key-countt,value-comnrevr

                    if (n1, n2) not in edgecount:
                        edgecount[(n1, n2)] = 0
                        G.add_edge(n1, n2)
                        edgedetails[(n1, n2)] = crlist


                    if (n1, n2) not in cmnrevrsedges[maincount]:
                        cmnrevrsedges[maincount].append((n1, n2))
                        edgecount[(n1, n2)] = edgecount[(n1, n2)] + 1



    A=nx.adjacency_matrix(G).todense()

    nx.write_gpickle(G, "../refine_groups/test.gpickle")

    return A, feature

def main():
    load_graph()



if __name__ == "__main__":
    main()