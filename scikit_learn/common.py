#import ROOT
import numpy as np
import pandas as pd
import math
import pickle

class DataGetter:

    def __init__(self, listToGet):
        self.listToGet = listToGet
        self.list2 = ["event." + v + "[i]" for v in self.listToGet]

        self.theStrCommand = "[" + ", ".join(self.list2) + "]"

    def getData(self, event, i):
        return eval(self.theStrCommand)

    def getList(self):
        return self.listToGet

def HEPReqs(event, i):
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
    m123 = event.cand_m[i]

    m12  = event.j12_m[i];
    m23  = event.j23_m[i];
    m13  = event.j13_m[i];
    dRMax = event.cand_dRMax[i]
    
    #HEP Pre requirements
    passPreRequirments = True
    passMassWindow = (minTopCandMass_ < m123) and (m123 < maxTopCandMass_)
    passPreRequirments = passMassWindow and dRMax < dRMax_

    #Implement HEP mass ratio requirements here
    criterionA = 0.2 < math.atan(m13/m12) and math.atan(m13/m12) < 1.3 and Rmin_ < m23/m123 and m23/m123 < Rmax_

    criterionB = ((Rmin_**2)*(1+(m13/m12)**2) < (1 - (m23/m123)**2)) and ((1 - (m23/m123)**2) < (Rmax_**2)*(1 + (m13/m12)**2))

    criterionC = ((Rmin_**2)*(1+(m12/m13)**2) < (1 - (m23/m123)**2)) and ((1 - (m23/m123)**2) < (Rmax_**2)*(1 + (m12/m13)**2))

    passHEPRequirments = criterionA or criterionB or criterionC;

    passBreq = (int(event.j1_CSV[i] > CSV_) + int(event.j2_CSV[i] > CSV_) + int(event.j3_CSV[i] > CSV_)) <= 1

    return passPreRequirments and passHEPRequirments and passBreq

class simpleTopCand:
    def __init__(self, event, i, discriminator):
        self.j1 = ROOT.TLorentzVector()
        self.j2 = ROOT.TLorentzVector()
        self.j3 = ROOT.TLorentzVector()
        self.j1.SetPtEtaPhiM(event.j1_pt[i], event.j1_eta[i], event.j1_phi[i], event.j1_m[i])
        self.j2.SetPtEtaPhiM(event.j2_pt[i], event.j2_eta[i], event.j2_phi[i], event.j2_m[i])
        self.j3.SetPtEtaPhiM(event.j3_pt[i], event.j3_eta[i], event.j3_phi[i], event.j3_m[i])
        self.discriminator = discriminator

    def __lt__(self, other):
        return self.discriminator < other.discriminator

def jetInList(jet, jlist):
    for j in jlist:
        if(abs(jet.M() - j.M()) < 0.0001):
            return True
    return False

def resolveOverlap(event, discriminators, threshold):
    topCands = [simpleTopCand(event, i, discriminators[i]) for i in xrange(len(event.j1_pt))]
    topCands.sort(reverse=True)

    finalTops = []
    usedJets = []
    for cand in topCands:
        #if not cand.j1 in usedJets and not cand.j2 in usedJets and not cand.j3 in usedJets:
        if not jetInList(cand.j1, usedJets) and not jetInList(cand.j2, usedJets) and not jetInList(cand.j3, usedJets):
            if cand.discriminator > threshold:
                usedJets += [cand.j1, cand.j2, cand.j3]
                finalTops.append(cand)

    return finalTops

class simpleTopCandHEP:
    def __init__(self, event, i, passFail):
        self.j1 = ROOT.TLorentzVector()
        self.j2 = ROOT.TLorentzVector()
        self.j3 = ROOT.TLorentzVector()
        self.j1.SetPtEtaPhiM(event.j1_pt[i], event.j1_eta[i], event.j1_phi[i], event.j1_m[i])
        self.j2.SetPtEtaPhiM(event.j2_pt[i], event.j2_eta[i], event.j2_phi[i], event.j2_m[i])
        self.j3.SetPtEtaPhiM(event.j3_pt[i], event.j3_eta[i], event.j3_phi[i], event.j3_m[i])
        self.cand_m = event.cand_m[i]
        self.passHEP = passFail

    def __lt__(self, other):
        return abs(self.cand_m - 173.4) < abs(other.cand_m - 173.4)

def resolveOverlapHEP(event, passFail):
    topCands = [simpleTopCandHEP(event, i, passFail[i]) for i in xrange(len(event.j1_pt))]
    topCands.sort(reverse=True)

    finalTops = []
    usedJets = []
    for cand in topCands:
        if not jetInList(cand.j1, usedJets) and not jetInList(cand.j2, usedJets) and not jetInList(cand.j3, usedJets):
            if cand.passHEP:
                usedJets += [cand.j1, cand.j2, cand.j3]
                finalTops.append(cand)

    return finalTops
