//go:build !cgo
// +build !cgo

// Package gpu provides GPU-accelerated Ringtail threshold signature operations.
// This file provides pure Go (stub) implementations when CGO is disabled.
package gpu

import (
	"errors"
	"sync"
)

// ErrCGODisabled is returned when GPU operations are called without CGO
var ErrCGODisabled = errors.New("CGO required for GPU-accelerated Ringtail operations")

// RingtailGPU provides GPU-accelerated operations for Ringtail threshold signing.
// This is a stub implementation when CGO is disabled.
type RingtailGPU struct {
	mu sync.RWMutex

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

// NewRingtailGPU creates a new GPU-accelerated Ringtail context (stub without CGO)
func NewRingtailGPU(cfg Config) (*RingtailGPU, error) {
	if cfg.N == 0 {
		cfg.N = 256
	}
	if cfg.Q == 0 {
		cfg.Q = 8380417
	}

	return &RingtailGPU{
		N:           cfg.N,
		Q:           cfg.Q,
		initialized: true,
	}, nil
}

// Close releases GPU resources (no-op without CGO)
func (r *RingtailGPU) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.initialized = false
}

// GPUAvailable returns false when CGO is disabled
func GPUAvailable() bool {
	return false
}

// GetBackend returns the active backend name
func GetBackend() string {
	return "CPU (CGO disabled)"
}

// =============================================================================
// NTT Operations (stubs)
// =============================================================================

// NTTForward returns error without CGO
func (r *RingtailGPU) NTTForward(polys [][]uint64) ([][]uint64, error) {
	return nil, ErrCGODisabled
}

// NTTInverse returns error without CGO
func (r *RingtailGPU) NTTInverse(polys [][]uint64) ([][]uint64, error) {
	return nil, ErrCGODisabled
}

// =============================================================================
// Polynomial Operations (stubs)
// =============================================================================

// PolyMulNTT returns error without CGO
func (r *RingtailGPU) PolyMulNTT(a, b []uint64) ([]uint64, error) {
	return nil, ErrCGODisabled
}

// PolyMul returns error without CGO
func (r *RingtailGPU) PolyMul(a, b [][]uint64) ([][]uint64, error) {
	return nil, ErrCGODisabled
}

// PolyAdd returns error without CGO
func (r *RingtailGPU) PolyAdd(a, b []uint64) ([]uint64, error) {
	return nil, ErrCGODisabled
}

// PolySub returns error without CGO
func (r *RingtailGPU) PolySub(a, b []uint64) ([]uint64, error) {
	return nil, ErrCGODisabled
}

// PolyScalarMul returns error without CGO
func (r *RingtailGPU) PolyScalarMul(a []uint64, scalar uint64) ([]uint64, error) {
	return nil, ErrCGODisabled
}

// =============================================================================
// Sampling Operations (stubs)
// =============================================================================

// SampleGaussian returns error without CGO
func (r *RingtailGPU) SampleGaussian(sigma float64, seed []byte) ([]uint64, error) {
	return nil, ErrCGODisabled
}

// SampleUniform returns error without CGO
func (r *RingtailGPU) SampleUniform(seed []byte) ([]uint64, error) {
	return nil, ErrCGODisabled
}

// SampleTernary returns error without CGO
func (r *RingtailGPU) SampleTernary(density float64, seed []byte) ([]uint64, error) {
	return nil, ErrCGODisabled
}

// =============================================================================
// Vector Operations (stubs)
// =============================================================================

// VectorNTTForward returns error without CGO
func (r *RingtailGPU) VectorNTTForward(vectors [][]uint64) ([][]uint64, error) {
	return nil, ErrCGODisabled
}

// VectorNTTInverse returns error without CGO
func (r *RingtailGPU) VectorNTTInverse(vectors [][]uint64) ([][]uint64, error) {
	return nil, ErrCGODisabled
}

// MatrixVectorMul returns error without CGO
func (r *RingtailGPU) MatrixVectorMul(matrix [][][]uint64, vector [][]uint64) ([][]uint64, error) {
	return nil, ErrCGODisabled
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

// RingtailGPUEnabled returns false when CGO is disabled
func RingtailGPUEnabled() bool {
	return false
}
