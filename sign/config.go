package sign

import "github.com/luxfi/ringtail/config"

// PARAMETERS - Default values, can be overridden by configuration
var (
	M               = 8
	N               = 7
	Dbar            = 48
	B               = 430070539612332.205811372782969  // 2^48.61156663661591
	Bsquare         = "184960669042442604975662780477" // B^2
	Kappa           = 23
	LogN            = 8
	SigmaE          = 6.108187070284607
	BoundE          = SigmaE * 2
	SigmaStar       = 172852667880.2713189548230532887787 // 2^37.33075191469097
	BoundStar       = SigmaStar * 2
	SigmaU          = 163961331.5239387
	BoundU          = SigmaU * 2
	KeySize         = 32              // 256 bits
	Q        uint64 = 0x1000000004A01 // 48-bit NTT-friendly prime
	QNu      uint64 = 0x80000
	QXi      uint64 = 0x40000
	TrustedDealerID = 0
	CombinerID      = 1
	Xi              = 30
	Nu              = 29
	EtaEpsilon      = 2.650104
)

// ApplyConfig updates parameters from configuration
func ApplyConfig(cfg *config.SignatureParams) {
	if cfg == nil {
		return
	}
	
	M = cfg.M
	N = cfg.N
	Dbar = cfg.Dbar
	B = cfg.B
	Kappa = cfg.Kappa
	LogN = cfg.LogN
	SigmaE = cfg.SigmaE
	BoundE = SigmaE * 2
	SigmaStar = cfg.SigmaStar
	BoundStar = SigmaStar * 2
	SigmaU = cfg.SigmaU
	BoundU = SigmaU * 2
	KeySize = cfg.KeySize
	Q = cfg.Q
	QNu = cfg.QNu
	QXi = cfg.QXi
	TrustedDealerID = cfg.TrustedDealerID
	CombinerID = cfg.CombinerID
	Xi = cfg.Xi
	Nu = cfg.Nu
	EtaEpsilon = cfg.EtaEpsilon
	K = cfg.PartyCount
	Threshold = cfg.Threshold
}
