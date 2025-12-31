// Package gpu provides parallel sampling operations for Ringtail threshold signatures.
// This file implements uniform, Gaussian, and ternary sampling with parallelization.
package gpu

import (
	"crypto/rand"
	"encoding/binary"
	"math"
	"sync"

	"github.com/zeebo/blake3"
)

// =============================================================================
// Batch Sampling Methods
// =============================================================================

// SampleUniform generates count uniform random polynomials in parallel.
// Each coefficient is uniformly distributed in [0, Q-1].
func (b *BatchNTT) SampleUniform(count int) [][]uint64 {
	if !b.initialized || count <= 0 {
		return nil
	}

	result := make([][]uint64, count)

	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			result[idx] = b.sampleUniformSingle()
		}(i)
	}
	wg.Wait()

	return result
}

// SampleGaussian generates count Gaussian-distributed polynomials in parallel.
// Uses Box-Muller transform with rejection sampling for discrete Gaussian.
// sigma is the standard deviation.
func (b *BatchNTT) SampleGaussian(count int, sigma float64) [][]uint64 {
	if !b.initialized || count <= 0 || sigma <= 0 {
		return nil
	}

	result := make([][]uint64, count)

	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			result[idx] = b.sampleGaussianSingle(sigma)
		}(i)
	}
	wg.Wait()

	return result
}

// SampleTernary generates count ternary polynomials in parallel.
// Each coefficient is in {-1, 0, 1} where -1 is represented as Q-1.
// Uses balanced ternary distribution: P(-1) = P(1) = 1/3, P(0) = 1/3
func (b *BatchNTT) SampleTernary(count int) [][]uint64 {
	if !b.initialized || count <= 0 {
		return nil
	}

	result := make([][]uint64, count)

	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			result[idx] = b.sampleTernarySingle()
		}(i)
	}
	wg.Wait()

	return result
}

// SampleUniformSeeded generates count uniform random polynomials using a seed.
// Deterministic: same seed produces same output.
func (b *BatchNTT) SampleUniformSeeded(count int, seed []byte) [][]uint64 {
	if !b.initialized || count <= 0 {
		return nil
	}

	result := make([][]uint64, count)

	// Use BLAKE3 XOF for deterministic random generation
	hasher := blake3.New()
	hasher.Write(seed)
	digest := hasher.Digest()

	for i := 0; i < count; i++ {
		result[i] = make([]uint64, b.N)
		buf := make([]byte, 8)
		for j := uint32(0); j < b.N; j++ {
			digest.Read(buf)
			// Rejection sampling for uniform distribution mod Q
			for {
				val := binary.LittleEndian.Uint64(buf) % (b.Q << 1) // Reduce bias
				if val < b.Q {
					result[i][j] = val
					break
				}
				digest.Read(buf)
			}
		}
	}

	return result
}

// SampleGaussianSeeded generates count Gaussian polynomials using a seed.
// Deterministic: same seed produces same output.
func (b *BatchNTT) SampleGaussianSeeded(count int, sigma float64, seed []byte) [][]uint64 {
	if !b.initialized || count <= 0 || sigma <= 0 {
		return nil
	}

	result := make([][]uint64, count)

	hasher := blake3.New()
	hasher.Write(seed)
	digest := hasher.Digest()

	tailBound := int64(sigma * 6) // 6-sigma bound

	for i := 0; i < count; i++ {
		result[i] = make([]uint64, b.N)
		for j := uint32(0); j < b.N; j++ {
			result[i][j] = sampleDiscreteGaussianXOF(digest, sigma, tailBound, b.Q)
		}
	}

	return result
}

// SampleTernarySeeded generates count ternary polynomials using a seed.
// Deterministic: same seed produces same output.
func (b *BatchNTT) SampleTernarySeeded(count int, seed []byte) [][]uint64 {
	if !b.initialized || count <= 0 {
		return nil
	}

	result := make([][]uint64, count)

	hasher := blake3.New()
	hasher.Write(seed)
	digest := hasher.Digest()

	buf := make([]byte, 1)
	for i := 0; i < count; i++ {
		result[i] = make([]uint64, b.N)
		for j := uint32(0); j < b.N; j++ {
			// Sample from {-1, 0, 1} with equal probability
			for {
				digest.Read(buf)
				val := buf[0] % 4 // 0, 1, 2, 3
				if val < 3 {      // Reject 3 to get uniform over {0, 1, 2}
					switch val {
					case 0:
						result[i][j] = b.Q - 1 // -1 mod Q
					case 1:
						result[i][j] = 0
					case 2:
						result[i][j] = 1
					}
					break
				}
			}
		}
	}

	return result
}

// =============================================================================
// Single Sample Methods (internal)
// =============================================================================

// sampleUniformSingle generates a single uniform random polynomial
func (b *BatchNTT) sampleUniformSingle() []uint64 {
	poly := make([]uint64, b.N)
	buf := make([]byte, 8)

	for i := uint32(0); i < b.N; i++ {
		// Rejection sampling for uniform mod Q
		for {
			rand.Read(buf)
			val := binary.LittleEndian.Uint64(buf)
			// To reduce bias, accept only if val < largest multiple of Q that fits
			maxValid := (^uint64(0) / b.Q) * b.Q
			if val < maxValid {
				poly[i] = val % b.Q
				break
			}
		}
	}

	return poly
}

// sampleGaussianSingle generates a single Gaussian polynomial using CDT
func (b *BatchNTT) sampleGaussianSingle(sigma float64) []uint64 {
	poly := make([]uint64, b.N)
	tailBound := int64(sigma * 6) // 6-sigma tail bound

	for i := uint32(0); i < b.N; i++ {
		poly[i] = sampleDiscreteGaussian(sigma, tailBound, b.Q)
	}

	return poly
}

// sampleTernarySingle generates a single ternary polynomial
func (b *BatchNTT) sampleTernarySingle() []uint64 {
	poly := make([]uint64, b.N)
	buf := make([]byte, 1)

	for i := uint32(0); i < b.N; i++ {
		// Sample from {-1, 0, 1} uniformly
		for {
			rand.Read(buf)
			val := buf[0] % 4
			if val < 3 {
				switch val {
				case 0:
					poly[i] = b.Q - 1 // -1 mod Q
				case 1:
					poly[i] = 0
				case 2:
					poly[i] = 1
				}
				break
			}
		}
	}

	return poly
}

// =============================================================================
// Discrete Gaussian Sampling
// =============================================================================

// sampleDiscreteGaussian samples from a discrete Gaussian distribution.
// Uses rejection sampling with tail bound.
func sampleDiscreteGaussian(sigma float64, tailBound int64, Q uint64) uint64 {
	// Precompute constants
	sigmaSquared2 := 2.0 * sigma * sigma
	normFactor := 1.0 / (sigma * math.Sqrt(2.0*math.Pi))

	buf := make([]byte, 16)
	for {
		// Sample candidate uniformly from [-tailBound, tailBound]
		rand.Read(buf)
		candidate := int64(binary.LittleEndian.Uint64(buf[:8]))%(2*tailBound+1) - tailBound

		// Compute acceptance probability
		prob := normFactor * math.Exp(-float64(candidate*candidate)/sigmaSquared2)

		// Sample uniform [0, 1)
		rand.Read(buf[8:])
		uniform := float64(binary.LittleEndian.Uint64(buf[8:])&((1<<53)-1)) / float64(1<<53)

		// Accept with probability proportional to Gaussian
		if uniform < prob*sigma*math.Sqrt(2.0*math.Pi) {
			// Convert to positive representation mod Q
			if candidate < 0 {
				return Q - uint64(-candidate)
			}
			return uint64(candidate)
		}
	}
}

// sampleDiscreteGaussianXOF samples using XOF (deterministic)
func sampleDiscreteGaussianXOF(xof *blake3.Digest, sigma float64, tailBound int64, Q uint64) uint64 {
	sigmaSquared2 := 2.0 * sigma * sigma
	normFactor := 1.0 / (sigma * math.Sqrt(2.0*math.Pi))

	buf := make([]byte, 16)
	for {
		xof.Read(buf)
		candidate := int64(binary.LittleEndian.Uint64(buf[:8]))%(2*tailBound+1) - tailBound

		prob := normFactor * math.Exp(-float64(candidate*candidate)/sigmaSquared2)

		uniform := float64(binary.LittleEndian.Uint64(buf[8:])&((1<<53)-1)) / float64(1<<53)

		if uniform < prob*sigma*math.Sqrt(2.0*math.Pi) {
			if candidate < 0 {
				return Q - uint64(-candidate)
			}
			return uint64(candidate)
		}
	}
}

// =============================================================================
// Specialized Sampling for Ringtail
// =============================================================================

// SampleSecretKey samples a secret key polynomial with small coefficients.
// Uses centered binomial distribution with parameter eta.
func (b *BatchNTT) SampleSecretKey(eta int) []uint64 {
	if !b.initialized || eta <= 0 {
		return nil
	}

	poly := make([]uint64, b.N)
	bytesNeeded := (eta + 7) / 8 * 2

	for i := uint32(0); i < b.N; i++ {
		buf := make([]byte, bytesNeeded*2)
		rand.Read(buf)

		// Centered binomial: sum of eta uniform bits minus sum of eta uniform bits
		sum := 0
		for j := 0; j < eta; j++ {
			byteIdx := j / 8
			bitIdx := j % 8
			if (buf[byteIdx]>>bitIdx)&1 == 1 {
				sum++
			}
			if (buf[bytesNeeded+byteIdx]>>bitIdx)&1 == 1 {
				sum--
			}
		}

		// Convert to positive representation mod Q
		if sum < 0 {
			poly[i] = b.Q - uint64(-sum)
		} else {
			poly[i] = uint64(sum)
		}
	}

	return poly
}

// SampleError samples an error polynomial with Gaussian distribution.
// Uses the sigma parameter from Ringtail's security parameters.
func (b *BatchNTT) SampleError(sigma float64) []uint64 {
	return b.sampleGaussianSingle(sigma)
}

// SampleMask samples a masking polynomial uniformly at random.
// Returns polynomial with all coefficients in [0, Q-1].
func (b *BatchNTT) SampleMask() []uint64 {
	return b.sampleUniformSingle()
}

// =============================================================================
// Batch Operations with Seeds
// =============================================================================

// ExpandSeed expands a seed into multiple random polynomials.
// Used for deriving public parameters from a compact seed.
func (b *BatchNTT) ExpandSeed(seed []byte, count int) [][]uint64 {
	return b.SampleUniformSeeded(count, seed)
}

// SampleMatrix samples a random matrix of polynomials.
// Returns rows x cols matrix with uniform random polynomials.
func (b *BatchNTT) SampleMatrix(rows, cols int, seed []byte) *GPUMatrix {
	if !b.initialized || rows <= 0 || cols <= 0 {
		return nil
	}

	// Generate row*cols polynomials
	polys := b.SampleUniformSeeded(rows*cols, seed)
	if polys == nil {
		return nil
	}

	return NewGPUMatrixFromData(uint32(rows), uint32(cols), polys, b, false)
}

// SampleVector samples a random vector of polynomials.
func (b *BatchNTT) SampleVector(length int, seed []byte) [][]uint64 {
	return b.SampleUniformSeeded(length, seed)
}

// SampleGaussianVector samples a vector of Gaussian polynomials.
func (b *BatchNTT) SampleGaussianVector(length int, sigma float64, seed []byte) [][]uint64 {
	return b.SampleGaussianSeeded(length, sigma, seed)
}
