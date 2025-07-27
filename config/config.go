package config

import (
	"encoding/json"
	"os"
)

// Config holds all configurable parameters for the signature scheme
type Config struct {
	// Network parameters
	NetworkConfig NetworkConfig `json:"network"`
	
	// Signature scheme parameters
	SignatureParams SignatureParams `json:"signature"`
	
	// Consensus integration
	ConsensusConfig ConsensusConfig `json:"consensus"`
}

// NetworkConfig holds network-related configuration
type NetworkConfig struct {
	ListenPort    int      `json:"listen_port"`
	PeerAddresses []string `json:"peer_addresses"`
	Timeout       int      `json:"timeout_seconds"`
	MaxRetries    int      `json:"max_retries"`
}

// SignatureParams holds signature scheme parameters
type SignatureParams struct {
	M               int     `json:"m"`
	N               int     `json:"n"`
	Dbar            int     `json:"dbar"`
	B               float64 `json:"b"`
	Kappa           int     `json:"kappa"`
	LogN            int     `json:"log_n"`
	SigmaE          float64 `json:"sigma_e"`
	SigmaStar       float64 `json:"sigma_star"`
	SigmaU          float64 `json:"sigma_u"`
	KeySize         int     `json:"key_size"`
	Q               uint64  `json:"q"`
	QNu             uint64  `json:"q_nu"`
	QXi             uint64  `json:"q_xi"`
	TrustedDealerID int     `json:"trusted_dealer_id"`
	CombinerID      int     `json:"combiner_id"`
	Xi              int     `json:"xi"`
	Nu              int     `json:"nu"`
	EtaEpsilon      float64 `json:"eta_epsilon"`
	PartyCount      int     `json:"party_count"`
	Threshold       int     `json:"threshold"`
}

// ConsensusConfig holds Lux consensus integration parameters
type ConsensusConfig struct {
	Enabled           bool   `json:"enabled"`
	ConsensusTimeout  int    `json:"consensus_timeout_ms"`
	SignatureTimeout  int    `json:"signature_timeout_ms"`
	MaxConcurrentSigs int    `json:"max_concurrent_signatures"`
	NodeID            string `json:"node_id"`
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		NetworkConfig: NetworkConfig{
			ListenPort:    8080,
			PeerAddresses: []string{},
			Timeout:       30,
			MaxRetries:    3,
		},
		SignatureParams: SignatureParams{
			M:               8,
			N:               7,
			Dbar:            48,
			B:               430070539612332.205811372782969,
			Kappa:           23,
			LogN:            8,
			SigmaE:          6.108187070284607,
			SigmaStar:       172852667880.2713189548230532887787,
			SigmaU:          163961331.5239387,
			KeySize:         32,
			Q:               0x1000000004A01,
			QNu:             0x80000,
			QXi:             0x40000,
			TrustedDealerID: 0,
			CombinerID:      1,
			Xi:              30,
			Nu:              29,
			EtaEpsilon:      2.650104,
			PartyCount:      3,
			Threshold:       3,
		},
		ConsensusConfig: ConsensusConfig{
			Enabled:           false,
			ConsensusTimeout:  5000,
			SignatureTimeout:  2000,
			MaxConcurrentSigs: 10,
			NodeID:            "",
		},
	}
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(path string) (*Config, error) {
	config := DefaultConfig()
	
	if path == "" {
		return config, nil
	}
	
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(config); err != nil {
		return nil, err
	}
	
	return config, nil
}

// SaveConfig saves configuration to a JSON file
func (c *Config) SaveConfig(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(c)
}