import FWCore.ParameterSet.Config as cms

GlobalMuonRefitter = cms.PSet(
    Chi2ProbabilityCut = cms.double(30.0),
    Direction = cms.int32(0),
    Chi2CutCSC = cms.double(150.0),
    HitThreshold = cms.int32(1),
    MuonHitsOption = cms.int32(1),
    Chi2CutRPC = cms.double(1.0),

    # muon station to be skipped
    SkipStation		= cms.int32(-1),

    # PXB = 1, PXF = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6
    TrackerSkipSystem	= cms.int32(-1),

    # layer, wheel, or disk depending on the system
    TrackerSkipSection	= cms.int32(-1),

    Fitter = cms.string('KFFitterForRefitInsideOut'),
    DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
    TrackerRecHitBuilder = cms.string('WithTrackAngle'),
    DoPredictionsOnly = cms.bool(False),
    Smoother = cms.string('KFSmootherForRefitInsideOut'),
    MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    RefitDirection = cms.string('insideOut'),
    CSCRecSegmentLabel = cms.InputTag("cscSegments"),
    RefitRPCHits = cms.bool(True),
    Chi2CutDT = cms.double(10.0),
    PtCut = cms.double(1.0),
    RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
    Propagator = cms.string('SmartPropagatorAnyRK')
)

