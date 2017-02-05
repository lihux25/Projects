import numpy as np
import pandas as pd
import math
import pickle

def baseTaggerReqs(row):
    Mw = 80.385
    Mt = 173.5
    Rmin_ = 0.85 *(Mw/Mt)
    Rmax_ = 1.25 *(Mw/Mt)
    CSV_ = 0.800
    minTopCandMass_ = 100
    maxTopCandMass_ = 250
    dRMax_ = 1.5

    #HEP tagger requirements
    passHEPRequirments = True

    #Get the total candidate mass
    m123 = row.cand_m

    m12  = row.j12_m;
    m23  = row.j23_m;
    m13  = row.j13_m;
    dRMax = row.cand_dRMax
    
    #HEP Pre requirements
    passPreRequirments = True
    passMassWindow = (minTopCandMass_ < m123) and (m123 < maxTopCandMass_)
    passPreRequirments = passMassWindow and dRMax < dRMax_

    #Implement HEP mass ratio requirements here
    criterionA = 0.2 < math.atan(m13/m12) and math.atan(m13/m12) < 1.3 and Rmin_ < m23/m123 and m23/m123 < Rmax_

    criterionB = ((Rmin_**2)*(1+(m13/m12)**2) < (1 - (m23/m123)**2)) and ((1 - (m23/m123)**2) < (Rmax_**2)*(1 + (m13/m12)**2))

    criterionC = ((Rmin_**2)*(1+(m12/m13)**2) < (1 - (m23/m123)**2)) and ((1 - (m23/m123)**2) < (Rmax_**2)*(1 + (m12/m13)**2))

    passHEPRequirments = criterionA or criterionB or criterionC;

    passBreq = (int(row.j1_CSV > CSV_) + int(row.j2_CSV > CSV_) + int(row.j3_CSV > CSV_)) <= 1

    return passPreRequirments and passHEPRequirments and passBreq

class simpleTopCand:
    def __init__(self, row):
        self.j1 = (row.j1_pt, row.j1_eta, row.j1_phi, row.j1_m)
        self.j2 = (row.j2_pt, row.j2_eta, row.j2_phi, row.j2_m)
        self.j3 = (row.j3_pt, row.j3_eta, row.j3_phi, row.j3_m)
        self.cand_m = row.cand_m
        self.disc = row.disc
        self.uniq_evtNum = row.evtNum
        self.uniq_procTypes = row.procTypes

    def __lt__(self, other):
        return self.disc < other.disc

def jetInList(jet, jlist):
    for j in jlist:
        if(abs(jet[-1] - j[-1]) < 0.0001):
            return True
    return False

def resolveOverlap(rows, threshold):
    topCands = [simpleTopCand(rows.iloc[i]) for i in range(len(rows))]
    topCands.sort(reverse=True)

    finalTops = []
    usedJets = []
    for cand in topCands:
        #if not cand.j1 in usedJets and not cand.j2 in usedJets and not cand.j3 in usedJets:
        if not jetInList(cand.j1, usedJets) and not jetInList(cand.j2, usedJets) and not jetInList(cand.j3, usedJets):
            if cand.disc > threshold:
                usedJets += [cand.j1, cand.j2, cand.j3]
                finalTops.append(cand)

    return finalTops

class simpleTopCandHEP:
    def __init__(self, row):
        self.j1 = (row.j1_pt, row.j1_eta, row.j1_phi, row.j1_m)
        self.j2 = (row.j2_pt, row.j2_eta, row.j2_phi, row.j2_m)
        self.j3 = (row.j3_pt, row.j3_eta, row.j3_phi, row.j3_m)
        self.cand_m = row.cand_m
        self.passHEP = row.passBaseTagger
        self.uniq_evtNum = row.evtNum
        self.uniq_procTypes = row.procTypes

    def __lt__(self, other):
        return abs(self.cand_m - 173.4) < abs(other.cand_m - 173.4)

def resolveOverlapHEP(rows):
    topCands = [simpleTopCandHEP(rows.iloc[i]) for i in range(len(rows))]
    topCands.sort(reverse=True)

    finalTops = []
    usedJets = []
    for cand in topCands:
        if not jetInList(cand.j1, usedJets) and not jetInList(cand.j2, usedJets) and not jetInList(cand.j3, usedJets):
            if cand.passHEP:
                usedJets += [cand.j1, cand.j2, cand.j3]
                finalTops.append(cand)

    return finalTops
