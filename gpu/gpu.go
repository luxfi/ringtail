//go:build !cgo
// +build !cgo

// Package gpu provides GPU-accelerated Ringtail threshold signature operations.
// This file provides pure Go implementations using the lattice/ring package when CGO is disabled.
package gpu

import (
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"sync"

	"github.com/luxfi/lattice/v7/ring"
	"github.com/luxfi/lattice/v7/utils/sampling"
)

// RingtailGPU provides operations for Ringtail threshold signing using pure Go.
type RingtailGPU struct {
	mu sync.RWMutex

	// Ring context from lattice package
	subRing *ring.SubRing

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

// NewRingtailGPU creates a new Ringtail context using pure Go
func NewRingtailGPU(cfg Config) (*RingtailGPU, error) {
	if cfg.N == 0 {
		cfg.N = 256
	}
	if cfg.Q == 0 {
		cfg.Q = 8380417
	}

	// Create SubRing with NTT support
	subRing, err := ring.NewSubRing(int(cfg.N), cfg.Q)
	if err != nil {
		return nil, fmt.Errorf("failed to create SubRing: %w", err)
	}

	// Generate NTT constants
	if err := subRing.GenNTTConstants(); err != nil {
		return nil, fmt.Errorf("failed to generate NTT constants: %w", err)
	}

	return &RingtailGPU{
		subRing:     subRing,
		N:           cfg.N,
		Q:           cfg.Q,
		initialized: true,
	}, nil
}

// Close releases resources (no-op for pure Go)
func (r *RingtailGPU) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.subRing = nil
	r.initialized = false
}

// GPUAvailable returns false when CGO is disabled
func GPUAvailable() bool {
	return false
}

// GetBackend returns the active backend name
func GetBackend() string {
	return "CPU (pure Go)"
}

// =============================================================================
// NTT Operations (pure Go via lattice/ring)
// =============================================================================

// NTTForward computes the forward NTT of polynomials
func (r *RingtailGPU) NTTForward(polys [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized || r.subRing == nil {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	results := make([][]uint64, len(polys))
	N := int(r.N)

	for i, poly := range polys {
		if len(poly) != N {
			return nil, fmt.Errorf("polynomial %d has wrong size: %d vs %d", i, len(poly), N)
		}

		results[i] = make([]uint64, N)
		copy(results[i], poly)
		r.subRing.NTT(results[i], results[i])
	}

	return results, nil
}

// NTTInverse computes the inverse NTT of polynomials
func (r *RingtailGPU) NTTInverse(polys [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized || r.subRing == nil {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	results := make([][]uint64, len(polys))
	N := int(r.N)

	for i, poly := range polys {
		if len(poly) != N {
			return nil, fmt.Errorf("polynomial %d has wrong size: %d vs %d", i, len(poly), N)
		}

		results[i] = make([]uint64, N)
		copy(results[i], poly)
		r.subRing.INTT(results[i], results[i])
	}

	return results, nil
}

// =============================================================================
// Polynomial Operations (pure Go)
// =============================================================================

// PolyMulNTT multiplies two polynomials in NTT domain (element-wise Hadamard product)
func (r *RingtailGPU) PolyMulNTT(a, b []uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized || r.subRing == nil {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	N := int(r.N)
	if len(a) != N || len(b) != N {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	result := make([]uint64, N)
	r.subRing.MulCoeffsMontgomery(a, b, result)

	return result, nil
}

// PolyMul multiplies batches of polynomials (handles NTT conversion internally)
func (r *RingtailGPU) PolyMul(a, b [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized || r.subRing == nil {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	if len(a) != len(b) {
		return nil, fmt.Errorf("batch size mismatch")
	}

	N := int(r.N)
	results := make([][]uint64, len(a))
	tempA := make([]uint64, N)
	tempB := make([]uint64, N)

	for i := range a {
		if len(a[i]) != N || len(b[i]) != N {
			return nil, fmt.Errorf("polynomial %d has wrong size", i)
		}

		results[i] = make([]uint64, N)

		// Copy and transform
		copy(tempA, a[i])
		copy(tempB, b[i])
		r.subRing.NTT(tempA, tempA)
		r.subRing.NTT(tempB, tempB)

		// Pointwise multiply
		r.subRing.MulCoeffsMontgomery(tempA, tempB, results[i])

		// Inverse transform
		r.subRing.INTT(results[i], results[i])
	}

	return results, nil
}

// PolyAdd adds two polynomials
func (r *RingtailGPU) PolyAdd(a, b []uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	if len(a) != len(b) {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	result := make([]uint64, len(a))
	Q := r.Q
	for i := range a {
		sum := a[i] + b[i]
		if sum >= Q {
			sum -= Q
		}
		result[i] = sum
	}

	return result, nil
}

// PolySub subtracts two polynomials
func (r *RingtailGPU) PolySub(a, b []uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	if len(a) != len(b) {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	result := make([]uint64, len(a))
	Q := r.Q
	for i := range a {
		if a[i] >= b[i] {
			result[i] = a[i] - b[i]
		} else {
			result[i] = Q - b[i] + a[i]
		}
	}

	return result, nil
}

// PolyScalarMul multiplies a polynomial by a scalar
func (r *RingtailGPU) PolyScalarMul(a []uint64, scalar uint64) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	result := make([]uint64, len(a))
	Q := r.Q
	for i := range a {
		prod := new(big.Int).SetUint64(a[i])
		prod.Mul(prod, new(big.Int).SetUint64(scalar))
		prod.Mod(prod, new(big.Int).SetUint64(Q))
		result[i] = prod.Uint64()
	}

	return result, nil
}

// =============================================================================
// Sampling Operations (pure Go)
// =============================================================================

// SampleGaussian samples a polynomial from a discrete Gaussian distribution
func (r *RingtailGPU) SampleGaussian(sigma float64, seed []byte) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	result := make([]uint64, r.N)
	Q := r.Q

	// Create PRNG
	var prng sampling.PRNG
	if len(seed) >= 32 {
		prng, _ = sampling.NewKeyedPRNG(seed[:32])
	} else {
		prng = sampling.NewPRNG()
	}

	bound := int64(math.Ceil(sigma * 6))

	for i := uint32(0); i < r.N; i++ {
		// Rejection sampling
		for {
			b := make([]byte, 8)
			prng.Read(b)
			sample := int64(b[0]) | int64(b[1])<<8 | int64(b[2])<<16 | int64(b[3])<<24
			sample = sample % (2*bound + 1) - bound

			prob := math.Exp(-float64(sample*sample) / (2 * sigma * sigma))
			threshold := make([]byte, 1)
			prng.Read(threshold)

			if float64(threshold[0])/256.0 < prob {
				if sample >= 0 {
					result[i] = uint64(sample)
				} else {
					result[i] = Q - uint64(-sample)
				}
				break
			}
		}
	}

	return result, nil
}

// SampleUniform samples a polynomial uniformly at random
func (r *RingtailGPU) SampleUniform(seed []byte) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	result := make([]uint64, r.N)
	Q := r.Q

	var prng sampling.PRNG
	if len(seed) >= 32 {
		prng, _ = sampling.NewKeyedPRNG(seed[:32])
	} else {
		prng = sampling.NewPRNG()
	}

	qBig := new(big.Int).SetUint64(Q)
	for i := uint32(0); i < r.N; i++ {
		b := make([]byte, 8)
		prng.Read(b)
		val := new(big.Int).SetBytes(b)
		val.Mod(val, qBig)
		result[i] = val.Uint64()
	}

	return result, nil
}

// SampleTernary samples a polynomial with ternary coefficients (-1, 0, 1)
func (r *RingtailGPU) SampleTernary(density float64, seed []byte) ([]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	result := make([]uint64, r.N)
	Q := r.Q

	var reader interface{ Read([]byte) (int, error) }
	if len(seed) >= 32 {
		prng, _ := sampling.NewKeyedPRNG(seed[:32])
		reader = prng
	} else {
		reader = rand.Reader
	}

	for i := uint32(0); i < r.N; i++ {
		b := make([]byte, 2)
		reader.Read(b)

		p := float64(b[0]) / 256.0
		if p < density {
			if b[1]&1 == 0 {
				result[i] = 1
			} else {
				result[i] = Q - 1
			}
		} else {
			result[i] = 0
		}
	}

	return result, nil
}

// =============================================================================
// Vector Operations (pure Go batch operations)
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
func (r *RingtailGPU) MatrixVectorMul(matrix [][][]uint64, vector [][]uint64) ([][]uint64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized || r.subRing == nil {
		return nil, fmt.Errorf("RingtailGPU not initialized")
	}

	rows := len(matrix)
	cols := len(vector)

	if rows == 0 || cols == 0 {
		return nil, nil
	}

	N := int(r.N)
	result := make([][]uint64, rows)

	for i := 0; i < rows; i++ {
		result[i] = make([]uint64, N)

		for j := 0; j < cols && j < len(matrix[i]); j++ {
			// Pointwise multiply in NTT domain
			prod := make([]uint64, N)
			r.subRing.MulCoeffsMontgomery(matrix[i][j], vector[j], prod)

			// Add to result
			for k := 0; k < N; k++ {
				sum := result[i][k] + prod[k]
				if sum >= r.Q {
					sum -= r.Q
				}
				result[i][k] = sum
			}
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

// GetRingtailGPU returns the global Ringtail instance
func GetRingtailGPU() (*RingtailGPU, error) {
	globalRingtailGPUOnce.Do(func() {
		globalRingtailGPU, globalRingtailGPUErr = NewRingtailGPU(DefaultConfig())
	})
	return globalRingtailGPU, globalRingtailGPUErr
}

// RingtailGPUEnabled returns false when CGO is disabled
func RingtailGPUEnabled() bool {
	return false
}
