from common import *
import sys
import os

NEVTS = {'ttbar':-1, 'zinv':-1}
#NEVTS = {'ttbar':500, 'zinv':500}
samplesToRun = {"ttbar":"trainingTuple_division_0_TTbarSingleLep_training.root", "zinv":"trainingTuple_division_0_ZJetsToNuNu_training.root"}
#samplesToRun = {"ttbar":"trainingTuple_division_1_TTbarSingleLep_validation.root", "zinv":"trainingTuple_division_1_ZJetsToNuNu_validation.root"}
output_str = samplesToRun.values()[0][:-5].split('_')[-1]

listToGet = ["cand_m", "j12_m", "j13_m", "j23_m", "j1_p", "j2_p", "j3_p", "dTheta12", "dTheta23", "dTheta13", "j1_CSV", "j2_CSV", "j3_CSV", "j1_QGL", "j2_QGL", "j3_QGL", "cand_dRMax", "j1_pt", "j2_pt", "j3_pt", "j1_eta", "j2_eta", "j3_eta", "j1_phi", "j2_phi", "j3_phi", "j1_m", "j2_m", "j3_m", "genConstiuentMatchesVec", "genTopMatchesVec", "genConstMatchGenPtVec", "cand_pt"]
 
dg = DataGetter(listToGet)

def getNormHist(datasetName, forceRedo = False):
    if not ('.root' in datasetName):
       print('Expect ".root" in input {}'.format(datasetName))
       sys.exit(1)
    stripName = datasetName[:-5]
    if not os.path.isfile('normHist_' + stripName + '.root') or forceRedo:
        dataset = ROOT.TFile.Open(datasetName)
        rtfile = ROOT.TFile.Open('normHist_' + stripName + '.root', 'RECREATE')
        hPtMatch = ROOT.TH1D("hPtMatch" + stripName, "hPtMatch", 50, 0.0, 2000.0)
        hPtMatch.Sumw2()
        hPtNoMatch = ROOT.TH1D("hPtNoMatch" + stripName, "hPtNoMatch", 50, 0.0, 2000.0)
        hPtNoMatch.Sumw2()
        print('Filling out normalization histograms for {}'.format(stripName))
        for event in dataset.slimmedTuple:
            for i in xrange(len(event.genConstiuentMatchesVec)):
                if event.genConstiuentMatchesVec[i] == 3 and event.genTopMatchesVec[i]:
                    hPtMatch.Fill(event.cand_pt[i], event.sampleWgt)
                else:
                    hPtNoMatch.Fill(event.cand_pt[i], event.sampleWgt)
        rtfile.Write()
        rtfile.Close()
        dataset.Close()
    rtfile = ROOT.TFile.Open('normHist_' + stripName + '.root')
    hPtMatch = rtfile.Get("hPtMatch" + stripName)
    hPtNoMatch = rtfile.Get("hPtNoMatch" + stripName)
    return (hPtMatch, hPtNoMatch, rtfile)

inputData = []
inputAnswer = []
inputWgts = []
procTypes = []
evtNum = []
input_aux = []

for proc, datasetName in samplesToRun.items():
    hPtMatch, hPtNoMatch, rtfile = getNormHist(datasetName)
    dataset = ROOT.TFile.Open(datasetName)
    print('\nProcessing sample : {} with {} entries'.format(datasetName, dataset.slimmedTuple.GetEntries()))

    NEVTS[proc] = dataset.slimmedTuple.GetEntries() if NEVTS[proc] == -1 else NEVTS[proc]
    print('Filling out all the dataset for {} events...'.format(NEVTS[proc]))
    Nevts = 0
    for event in dataset.slimmedTuple:
        if Nevts >= NEVTS[proc]:
            break
        Nevts +=1
        if Nevts ==1 or Nevts%(NEVTS[proc]/10) == 1 or Nevts == NEVTS[proc]:
           print('  Processing the {}st event ... '.format(Nevts))
        for i in xrange(len(event.cand_m)):
            inputData.append(dg.getData(event, i))
            nmatch = event.genConstiuentMatchesVec[i]
            inputAnswer.append(int(nmatch == 3) and event.genTopMatchesVec[i])
            procTypes.append(proc)
            evtNum.append(Nevts)
            input_aux.append([event.MET, event.Njet, event.Bjet, event.sampleWgt])
            if int(nmatch == 3) and event.genTopMatchesVec[i]:
                if hPtMatch.GetBinContent(hPtMatch.FindBin(event.cand_pt[i])) > 10:
                    inputWgts.append(1.0 / hPtMatch.GetBinContent(hPtMatch.FindBin(event.cand_pt[i])))
                else:
                    inputWgts.append(0.0)
            else:
                if hPtNoMatch.GetBinContent(hPtNoMatch.FindBin(event.cand_pt[i])) > 10:
                    inputWgts.append(1.0 / hPtNoMatch.GetBinContent(hPtNoMatch.FindBin(event.cand_pt[i])))
                else:
                    inputWgts.append(0.0)

df_inputData = pd.DataFrame(inputData, columns=listToGet)
df_inputData['answer'] = inputAnswer
df_inputData['weight'] = inputWgts
df_inputData['procTypes'] = procTypes
df_inputData['evtNum'] = evtNum

df_aux = pd.DataFrame(input_aux, columns=['MET', 'Njet', 'Bjet', 'sampleWgt'])
df_inputData = df_inputData.join(df_aux)

df_inputData.to_csv(output_str+'.csv', compression='gzip')

print(df_inputData.iloc[:5])
print(df_inputData.iloc[-6:])
