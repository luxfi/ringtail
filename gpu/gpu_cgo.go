//go:build cgo
// +build cgo

// Package gpu provides GPU-accelerated Ringtail threshold signature operations.
// This package wraps the lattice/gpu package to provide high-performance
// NTT-based threshold signatures for post-quantum consensus.
package gpu

import (
	"fmt"
	"sync"

	latticegpu "github.com/luxfi/lattice/v6/gpu"
)

// RingtailGPU provides GPU-accelerated operations for Ringtail threshold signing.
// It uses the lattice/gpu package for NTT operations.
type RingtailGPU struct {
	mu sync.RWMutex

	// NTT context for the main ring
	nttCtx *latticegpu.NTTContext

	// Ring parameters
	N uint32 // Polynomial degree
	Q uint64 // Modulus

	// State
	initialized bool
}

// Config configures the GPU-accelerated Ringtail operations
type Config struct {
	// Ring parameters - must match the ringtail sign package config
	N uint32 // Polynomial degree (default: 256)
	Q uint64 // Modulus (default: Ringtail's Q)
}

// DefaultConfig returns the default configuration matching Ringtail's parameters
func DefaultConfig() Config {
	return Config{
		N: 256,
		Q: 8380417, // Ringtail's modulus
	}
}

// NewRingtailGPU creates a new GPU-accelerated Ringtail context
func NewRingtailGPU(cfg Config) (*RingtailGPU, error) {
	if cfg.N == 0 {
		cfg.N = 256
	}
	if cfg.Q == 0 {
		cfg.Q = 8380417
	}

	nttCtx, err := latticegpu.NewNTTContext(cfg.N, cfg.Q)
	if err != nil {
		return nil, fmt.Errorf("failed to create NTT context: %w", err)
	}

	return &RingtailGPU{
		nttCtx:      nttCtx,
		N:           cfg.N,
		Q:           cfg.Q,
		initialized: true,
	}, nil
}

// Close releases GPU resources
func (r *RingtailGPU) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.nttCtx != nil {
		r.nttCtx.Close()
		r.nttCtx = nil
	}
	r.initialized = false
}

// GPUAvailable returns true if GPU acceleration is available
func GPUAvailable() bool {
	return latticegpu.GPUAvailable()
}

// GetBackend returns the active backend name
func GetBackend() string {
	return latticegpu.GetBackend()
}

// =============================================================================
// NTT Operations (GPU-accelerated)
// =============================================================================

// NTTForward computes the forward NTT of polynomials (batch operation)
func (r *RingtailGPU) NTTForward(polys [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return r.nttCtx.NTT(polys)
}

// NTTInverse computes the inverse NTT of polynomials (batch operation)
func (r *RingtailGPU) NTTInverse(polys [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return r.nttCtx.INTT(polys)
}

// =============================================================================
// Polynomial Operations (GPU-accelerated)
// =============================================================================

// PolyMulNTT multiplies two polynomials in NTT domain (element-wise Hadamard product)
func (r *RingtailGPU) PolyMulNTT(a, b []uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return r.nttCtx.PolyMulNTT(a, b)
}

// PolyMul multiplies batches of polynomials (handles NTT conversion internally)
func (r *RingtailGPU) PolyMul(a, b [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return r.nttCtx.PolyMul(a, b)
}

// PolyAdd adds two polynomials
func (r *RingtailGPU) PolyAdd(a, b []uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return latticegpu.PolyAdd(a, b, r.Q)
}

// PolySub subtracts two polynomials
func (r *RingtailGPU) PolySub(a, b []uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return latticegpu.PolySub(a, b, r.Q)
}

// PolyScalarMul multiplies a polynomial by a scalar
func (r *RingtailGPU) PolyScalarMul(a []uint64, scalar uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return latticegpu.PolyScalarMul(a, scalar, r.Q)
}

// =============================================================================
// Sampling Operations (GPU-accelerated)
// =============================================================================

// SampleGaussian samples a polynomial from a discrete Gaussian distribution
func (r *RingtailGPU) SampleGaussian(sigma float64, seed []byte) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return latticegpu.SampleGaussian(r.N, r.Q, sigma, seed)
}

// SampleUniform samples a polynomial uniformly at random
func (r *RingtailGPU) SampleUniform(seed []byte) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return latticegpu.SampleUniform(r.N, r.Q, seed)
}

// SampleTernary samples a polynomial with ternary coefficients (-1, 0, 1)
func (r *RingtailGPU) SampleTernary(density float64, seed []byte) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	return latticegpu.SampleTernary(r.N, r.Q, density, seed)
}

// =============================================================================
// Vector Operations (GPU-accelerated batch operations)
// =============================================================================

// VectorNTTForward converts a vector of polynomials to NTT domain
func (r *RingtailGPU) VectorNTTForward(vectors [][]uint64) ([][]uint64, error) {
	return r.NTTForward(vectors)
}

// VectorNTTInverse converts a vector of polynomials from NTT domain
func (r *RingtailGPU) VectorNTTInverse(vectors [][]uint64) ([][]uint64, error) {
	return r.NTTInverse(vectors)
}

// MatrixVectorMul computes result = A * v where A is a matrix and v is a vector
// Both must be in NTT domain for efficient multiplication
func (r *RingtailGPU) MatrixVectorMul(matrix [][][]uint64, vector [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	rows := len(matrix)
	cols := len(vector)

	if rows == 0 || cols == 0 {
		return nil, nil
	}

	result := make([][]uint64, rows)

	// For each row in the matrix
	for i := 0; i < rows; i++ {
		// Initialize result[i] to zero
		result[i] = make([]uint64, r.N)

		// Compute dot product: result[i] = sum(matrix[i][j] * vector[j])
		for j := 0; j < cols && j < len(matrix[i]); j++ {
			// Multiply in NTT domain
			prod, err := r.nttCtx.PolyMulNTT(matrix[i][j], vector[j])
			if err != nil {
				return nil, fmt.Errorf("matrix-vector mul at [%d][%d]: %w", i, j, err)
			}

			// Add to result
			sum, err := latticegpu.PolyAdd(result[i], prod, r.Q)
			if err != nil {
				return nil, fmt.Errorf("matrix-vector add at [%d][%d]: %w", i, j, err)
			}
			result[i] = sum
		}
	}

	return result, nil
}

// =============================================================================
// Global Instance
// =============================================================================

var (
	globalRingtailGPU     *RingtailGPU
	globalRingtailGPUOnce sync.Once
	globalRingtailGPUErr  error
)

// GetRingtailGPU returns the global GPU-accelerated Ringtail instance
func GetRingtailGPU() (*RingtailGPU, error) {
	globalRingtailGPUOnce.Do(func() {
		globalRingtailGPU, globalRingtailGPUErr = NewRingtailGPU(DefaultConfig())
	})
	return globalRingtailGPU, globalRingtailGPUErr
}

// RingtailGPUEnabled returns true if GPU acceleration is available for Ringtail
func RingtailGPUEnabled() bool {
	return GPUAvailable()
}
