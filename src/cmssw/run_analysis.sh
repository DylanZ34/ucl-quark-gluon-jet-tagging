#!/bin/bash

set -e

NEVENTS=${1:-1000}
SKIP_EVENTS=${2:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

JET_FILE="/home/cmsusr/jets_${TIMESTAMP}.csv"
CONSTITUENT_FILE="/home/cmsusr/constituents_${TIMESTAMP}.csv"

echo "Running CMS Open Data analysis"
echo "Number of events: $NEVENTS"
echo "Skip events: $SKIP_EVENTS"
echo "Jet output file: $JET_FILE"
echo "Constituent output file: $CONSTITUENT_FILE"

cd /code/CMSSW_10_6_30/src

echo "Setting up CMS environment..."
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval "$(scram runtime -sh)"

echo "Writing DemoAnalyzer.cc..."

cat > Demo/DemoAnalyzer/plugins/DemoAnalyzer.cc << 'CPP'
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class DemoAnalyzer : public edm::EDAnalyzer {
public:
  explicit DemoAnalyzer(const edm::ParameterSet&);
  ~DemoAnalyzer();

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string getTruthLabel(int partonFlavour);

  edm::EDGetTokenT<pat::JetCollection> jetToken_;

  std::ofstream jetFile_;
  std::ofstream constituentFile_;
};

DemoAnalyzer::DemoAnalyzer(const edm::ParameterSet& iConfig) {
  jetToken_ = consumes<pat::JetCollection>(
      iConfig.getParameter<edm::InputTag>("jets")
  );

  std::string jetOutputFile =
      iConfig.getParameter<std::string>("jetOutputFile");

  std::string constituentOutputFile =
      iConfig.getParameter<std::string>("constituentOutputFile");

  jetFile_.open(jetOutputFile);
  constituentFile_.open(constituentOutputFile);

  jetFile_ << "run,lumi,event,jet_index,"
           << "jet_pt,jet_eta,jet_phi,jet_mass,n_constituents,"
           << "jet_parton_flavour,jet_hadron_flavour,truth_label\n";

  constituentFile_ << "run,lumi,event,jet_index,constituent_index,"
                   << "constituent_pt,constituent_eta,constituent_phi,"
                   << "constituent_mass,pdgId,charge,"
                   << "jet_parton_flavour,jet_hadron_flavour,truth_label\n";
}

DemoAnalyzer::~DemoAnalyzer() {
  jetFile_.close();
  constituentFile_.close();
}

std::string DemoAnalyzer::getTruthLabel(int partonFlavour) {
  int absFlavour = std::abs(partonFlavour);

  if (absFlavour >= 1 && absFlavour <= 6) {
    return "quark";
  }

  if (partonFlavour == 21) {
    return "gluon";
  }

  return "unknown";
}

void DemoAnalyzer::analyze(const edm::Event& iEvent,
                           const edm::EventSetup& iSetup) {
  edm::Handle<pat::JetCollection> jets;
  iEvent.getByToken(jetToken_, jets);

  std::cout << "Run: " << iEvent.id().run()
            << " Lumi: " << iEvent.luminosityBlock()
            << " Event: " << iEvent.id().event()
            << " Number of jets: " << jets->size()
            << std::endl;

  int jetIndex = 0;

  for (const auto& jet : *jets) {
    int nConstituents = jet.numberOfDaughters();

    int partonFlavour = jet.partonFlavour();
    int hadronFlavour = jet.hadronFlavour();

    std::string truthLabel = getTruthLabel(partonFlavour);

    jetFile_ << iEvent.id().run() << ","
             << iEvent.luminosityBlock() << ","
             << iEvent.id().event() << ","
             << jetIndex << ","
             << jet.pt() << ","
             << jet.eta() << ","
             << jet.phi() << ","
             << jet.mass() << ","
             << nConstituents << ","
             << partonFlavour << ","
             << hadronFlavour << ","
             << truthLabel << "\n";

    for (int i = 0; i < nConstituents; i++) {
      const reco::Candidate* c = jet.daughter(i);

      constituentFile_ << iEvent.id().run() << ","
                       << iEvent.luminosityBlock() << ","
                       << iEvent.id().event() << ","
                       << jetIndex << ","
                       << i << ","
                       << c->pt() << ","
                       << c->eta() << ","
                       << c->phi() << ","
                       << c->mass() << ","
                       << c->pdgId() << ","
                       << c->charge() << ","
                       << partonFlavour << ","
                       << hadronFlavour << ","
                       << truthLabel << "\n";
    }

    jetIndex++;
  }
}

DEFINE_FWK_MODULE(DemoAnalyzer);
CPP

echo "Writing ConfFile_cfg.py..."

cat > Demo/DemoAnalyzer/python/ConfFile_cfg.py << 'PY'
import FWCore.ParameterSet.Config as cms
import sys

nevents = 1000
skipevents = 0
jet_output_file = "/home/cmsusr/jets.csv"
constituent_output_file = "/home/cmsusr/constituents.csv"

for arg in sys.argv:
    if arg.startswith("nevents="):
        nevents = int(arg.split("=", 1)[1])
    elif arg.startswith("skipEvents="):
        skipevents = int(arg.split("=", 1)[1])
    elif arg.startswith("jetOutputFile="):
        jet_output_file = arg.split("=", 1)[1]
    elif arg.startswith("constituentOutputFile="):
        constituent_output_file = arg.split("=", 1)[1]

print("Processing", nevents, "events")
print("Skipping", skipevents, "events")
print("Jet output file:", jet_output_file)
print("Constituent output file:", constituent_output_file)

process = cms.Process("Demo")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nevents)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16MiniAODv2/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v2/2520000/0A47ADFB-CF87-A545-B45C-4821A69009B1.root"
    ),
    skipEvents = cms.untracked.uint32(skipevents)
)

process.demo = cms.EDAnalyzer(
    "DemoAnalyzer",
    jets = cms.InputTag("slimmedJets"),
    jetOutputFile = cms.string(jet_output_file),
    constituentOutputFile = cms.string(constituent_output_file)
)

process.p = cms.Path(process.demo)
PY

echo "Compiling analyzer..."
scram b

echo "Running cmsRun..."
cmsRun Demo/DemoAnalyzer/python/ConfFile_cfg.py \
  nevents=$NEVENTS \
  skipEvents=$SKIP_EVENTS \
  jetOutputFile=$JET_FILE \
  constituentOutputFile=$CONSTITUENT_FILE

echo "Done."
echo "Output files:"
ls -lh "$JET_FILE" "$CONSTITUENT_FILE"

echo "Preview of jets file:"
head "$JET_FILE"

echo "Truth label counts:"
python3 - << PYCOUNT
import pandas as pd
df = pd.read_csv("$JET_FILE")
print(df["truth_label"].value_counts(dropna=False))
print()
print("parton flavour counts:")
print(df["jet_parton_flavour"].value_counts(dropna=False).sort_index())
PYCOUNT

echo "Preview of constituents file:"
head "$CONSTITUENT_FILE"
