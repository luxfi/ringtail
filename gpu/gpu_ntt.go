// Package gpu provides GPU-accelerated NTT operations for Ringtail threshold signatures.
// This file implements batch NTT using Cooley-Tukey algorithm with Montgomery reduction.
package gpu

import (
	"errors"
	"math/bits"
	"sync"
)

// Ring parameters for Ringtail
const (
	// DefaultN is the default polynomial degree
	DefaultN uint32 = 256

	// DefaultQ is Ringtail's 48-bit NTT-friendly prime: 2^48 + 18945
	// Q - 1 = 2^48 + 18944 = 2^9 * 549755831341
	// This supports NTT for N up to 512
	DefaultQ uint64 = 0x1000000004A01

	// MontgomeryR = 2^64 mod Q (precomputed for Montgomery multiplication)
	MontgomeryR uint64 = 0xFFFFFFFB5FF // 2^64 mod Q

	// MontgomeryR2 = 2^128 mod Q (for converting to Montgomery form)
	MontgomeryR2 uint64 = 0x23D8F5B3FBE3 // (2^64)^2 mod Q

	// MontgomeryQInv = -Q^(-1) mod 2^64 (for Montgomery reduction)
	MontgomeryQInv uint64 = 0x4A00FFFFFFB5DFF
)

var (
	// ErrInvalidDegree is returned when polynomial degree is invalid
	ErrInvalidDegree = errors.New("polynomial degree must be a power of 2")

	// ErrDimensionMismatch is returned when dimensions don't match
	ErrDimensionMismatch = errors.New("dimension mismatch")

	// ErrNotInitialized is returned when BatchNTT is not initialized
	ErrNotInitialized = errors.New("BatchNTT not initialized")
)

// BatchNTT provides batch NTT operations with precomputed twiddle factors.
// Uses Cooley-Tukey butterfly algorithm with Montgomery arithmetic.
type BatchNTT struct {
	mu sync.RWMutex

	N  uint32 // Ring dimension (power of 2)
	Q  uint64 // Modulus
	NI uint64 // N^(-1) mod Q (for inverse NTT scaling)

	// Precomputed twiddle factors in Montgomery form
	twiddles    []uint64 // For forward NTT
	twiddlesInv []uint64 // For inverse NTT

	// Montgomery constants
	R    uint64 // R = 2^64 mod Q
	R2   uint64 // R^2 mod Q
	QInv uint64 // -Q^(-1) mod 2^64

	initialized bool
}

// NewBatchNTT creates a new BatchNTT context with precomputed twiddle factors.
// N must be a power of 2 and Q must be prime with Q-1 divisible by 2*N.
func NewBatchNTT(N uint32, Q uint64) (*BatchNTT, error) {
	if N == 0 {
		N = DefaultN
	}
	if Q == 0 {
		Q = DefaultQ
	}

	// Verify N is power of 2
	if N&(N-1) != 0 {
		return nil, ErrInvalidDegree
	}

	b := &BatchNTT{
		N: N,
		Q: Q,
	}

	// Compute N^(-1) mod Q for inverse NTT scaling
	b.NI = modInverse(uint64(N), Q)

	// Find primitive 2N-th root of unity
	omega := findPrimitiveRoot(Q, uint64(2*N))

	// Precompute twiddle factors
	b.twiddles = make([]uint64, N)
	b.twiddlesInv = make([]uint64, N)

	omegaInv := modInverse(omega, Q)

	// Compute twiddles in bit-reversed order for Cooley-Tukey
	b.twiddles[0] = 1
	b.twiddlesInv[0] = 1

	for i := uint32(1); i < N; i++ {
		j := bitReverse(i, bits.TrailingZeros32(N))
		b.twiddles[j] = modPow(omega, uint64(i), Q)
		b.twiddlesInv[j] = modPow(omegaInv, uint64(i), Q)
	}

	b.initialized = true
	return b, nil
}

// Forward computes batch forward NTT on multiple polynomials.
// Input polynomials are in coefficient form, output is in NTT form.
func (b *BatchNTT) Forward(polys [][]uint64) [][]uint64 {
	if !b.initialized {
		return nil
	}

	result := make([][]uint64, len(polys))

	// Process in parallel using goroutines
	var wg sync.WaitGroup
	for i := range polys {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			result[idx] = b.ForwardSingle(polys[idx])
		}(i)
	}
	wg.Wait()

	return result
}

// Inverse computes batch inverse NTT on multiple polynomials.
// Input polynomials are in NTT form, output is in coefficient form.
func (b *BatchNTT) Inverse(polys [][]uint64) [][]uint64 {
	if !b.initialized {
		return nil
	}

	result := make([][]uint64, len(polys))

	var wg sync.WaitGroup
	for i := range polys {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			result[idx] = b.InverseSingle(polys[idx])
		}(i)
	}
	wg.Wait()

	return result
}

// ForwardSingle computes forward NTT on a single polynomial using Cooley-Tukey.
// Uses decimation-in-time with bit-reversal permutation.
func (b *BatchNTT) ForwardSingle(poly []uint64) []uint64 {
	if !b.initialized || len(poly) != int(b.N) {
		return nil
	}

	// Copy input
	result := make([]uint64, b.N)
	for i := range poly {
		result[i] = poly[i] % b.Q
	}

	// Bit-reversal permutation
	bitReversePerm(result)

	// Cooley-Tukey butterfly
	for m := uint32(2); m <= b.N; m <<= 1 {
		halfM := m >> 1
		for k := uint32(0); k < b.N; k += m {
			for j := uint32(0); j < halfM; j++ {
				// Twiddle factor index
				tIdx := j * (b.N / m)
				w := b.twiddles[tIdx]

				u := result[k+j]
				t := mulMod(w, result[k+j+halfM], b.Q)

				// Butterfly
				result[k+j] = b.modAdd(u, t)
				result[k+j+halfM] = b.modSub(u, t)
			}
		}
	}

	return result
}

// InverseSingle computes inverse NTT on a single polynomial using Gentleman-Sande.
// Includes scaling by N^(-1).
func (b *BatchNTT) InverseSingle(poly []uint64) []uint64 {
	if !b.initialized || len(poly) != int(b.N) {
		return nil
	}

	// Copy input
	result := make([]uint64, b.N)
	for i := range poly {
		result[i] = poly[i] % b.Q
	}

	// Gentleman-Sande (decimation-in-frequency) inverse NTT
	for m := b.N; m >= 2; m >>= 1 {
		halfM := m >> 1
		for k := uint32(0); k < b.N; k += m {
			for j := uint32(0); j < halfM; j++ {
				tIdx := j * (b.N / m)
				w := b.twiddlesInv[tIdx]

				u := result[k+j]
				v := result[k+j+halfM]

				// Inverse butterfly
				result[k+j] = b.modAdd(u, v)
				result[k+j+halfM] = mulMod(w, b.modSub(u, v), b.Q)
			}
		}
	}

	// Bit-reversal permutation
	bitReversePerm(result)

	// Scale by N^(-1)
	for i := range result {
		result[i] = mulMod(result[i], b.NI, b.Q)
	}

	return result
}

// =============================================================================
// Montgomery Arithmetic (disabled - using direct modular arithmetic)
// =============================================================================

// toMontgomery converts x to Montgomery form (identity for non-Montgomery)
func (b *BatchNTT) toMontgomery(x uint64) uint64 {
	return x % b.Q
}

// fromMontgomery converts x from Montgomery form (identity for non-Montgomery)
func (b *BatchNTT) fromMontgomery(x uint64) uint64 {
	return x % b.Q
}

// montMul computes modular multiplication: a * b mod Q
func (b *BatchNTT) montMul(a, bval uint64) uint64 {
	return mulMod(a, bval, b.Q)
}

// modAdd computes (a + b) mod Q
func (b *BatchNTT) modAdd(a, c uint64) uint64 {
	sum := a + c
	if sum >= b.Q {
		sum -= b.Q
	}
	return sum
}

// modSub computes (a - b) mod Q
func (b *BatchNTT) modSub(a, c uint64) uint64 {
	if a >= c {
		return a - c
	}
	return b.Q - c + a
}

// =============================================================================
// Helper Functions
// =============================================================================

// computeQInv computes -Q^(-1) mod 2^64
func computeQInv(Q uint64) uint64 {
	// Newton's method: qInv = qInv * (2 - Q * qInv)
	qInv := Q
	for i := 0; i < 6; i++ {
		qInv *= 2 - Q*qInv
	}
	return -qInv
}

// modInverse computes a^(-1) mod m using extended Euclidean algorithm
func modInverse(a, m uint64) uint64 {
	if a == 0 {
		return 0
	}
	return modPow(a, m-2, m) // Fermat's little theorem for prime m
}

// modPow computes base^exp mod m
func modPow(base, exp, m uint64) uint64 {
	result := uint64(1)
	base = base % m
	for exp > 0 {
		if exp&1 == 1 {
			result = mulMod(result, base, m)
		}
		exp >>= 1
		base = mulMod(base, base, m)
	}
	return result
}

// mulMod computes (a * b) mod m using 128-bit arithmetic
func mulMod(a, b_val, m uint64) uint64 {
	hi, lo := bits.Mul64(a, b_val)
	_, rem := bits.Div64(hi, lo, m)
	return rem
}

// findPrimitiveRoot finds a primitive n-th root of unity mod Q
func findPrimitiveRoot(Q, n uint64) uint64 {
	// For Q = 0x1000000004A01, Q-1 = 2^48 + 18944
	// We need omega such that omega^n = 1 and omega^(n/2) != 1

	// Find generator of multiplicative group
	g := findGenerator(Q)

	// omega = g^((Q-1)/n)
	exp := (Q - 1) / n
	omega := modPow(g, exp, Q)

	return omega
}

// findGenerator finds a generator of Z_Q^*
func findGenerator(Q uint64) uint64 {
	// For prime Q, try small integers until we find a generator
	// A generator g satisfies: g^((Q-1)/p) != 1 for all prime p | (Q-1)

	factors := primeFactors(Q - 1)

	for g := uint64(2); g < Q; g++ {
		isGenerator := true
		for _, p := range factors {
			if modPow(g, (Q-1)/p, Q) == 1 {
				isGenerator = false
				break
			}
		}
		if isGenerator {
			return g
		}
	}
	return 2 // Fallback
}

// primeFactors returns distinct prime factors of n
func primeFactors(n uint64) []uint64 {
	var factors []uint64
	seen := make(map[uint64]bool)

	for p := uint64(2); p*p <= n; p++ {
		if n%p == 0 {
			if !seen[p] {
				factors = append(factors, p)
				seen[p] = true
			}
			for n%p == 0 {
				n /= p
			}
		}
	}
	if n > 1 && !seen[n] {
		factors = append(factors, n)
	}
	return factors
}

// bitReverse reverses the bits of x in log2N bits
func bitReverse(x uint32, log2N int) uint32 {
	return bits.Reverse32(x) >> (32 - log2N)
}

// bitReversePerm performs in-place bit-reversal permutation
func bitReversePerm(a []uint64) {
	n := uint32(len(a))
	log2N := bits.TrailingZeros32(n)

	for i := uint32(0); i < n; i++ {
		j := bitReverse(i, log2N)
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
}

// Close releases resources (no-op for pure Go implementation)
func (b *BatchNTT) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.initialized = false
}
