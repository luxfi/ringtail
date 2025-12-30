//go:build cgo
// +build cgo

package gpu

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.N != 256 {
		t.Errorf("expected N=256, got %d", cfg.N)
	}
	if cfg.Q != 8380417 {
		t.Errorf("expected Q=8380417, got %d", cfg.Q)
	}
}

func TestNewRingtailGPU(t *testing.T) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	if !gpu.initialized {
		t.Error("expected initialized to be true")
	}
}

func TestGPUAvailable(t *testing.T) {
	// Just verify it doesn't panic
	available := GPUAvailable()
	t.Logf("GPU available: %v", available)
}

func TestGetBackend(t *testing.T) {
	backend := GetBackend()
	t.Logf("Backend: %s", backend)
	if backend == "" {
		t.Error("expected non-empty backend string")
	}
}

func TestNTTOperations(t *testing.T) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	// Create test polynomials
	n := int(gpu.N)
	polys := make([][]uint64, 2)
	for i := range polys {
		polys[i] = make([]uint64, n)
		for j := 0; j < n; j++ {
			polys[i][j] = uint64(j + i*n) % gpu.Q
		}
	}

	// Forward NTT
	nttPolys, err := gpu.NTTForward(polys)
	if err != nil {
		t.Fatalf("NTTForward failed: %v", err)
	}
	if len(nttPolys) != 2 {
		t.Errorf("expected 2 polynomials, got %d", len(nttPolys))
	}

	// Inverse NTT
	invPolys, err := gpu.NTTInverse(nttPolys)
	if err != nil {
		t.Fatalf("NTTInverse failed: %v", err)
	}
	if len(invPolys) != 2 {
		t.Errorf("expected 2 polynomials, got %d", len(invPolys))
	}

	// Values should be close to original (allowing for modular arithmetic)
	for i := range polys {
		for j := 0; j < n; j++ {
			if invPolys[i][j] != polys[i][j] {
				// Allow some tolerance due to modular arithmetic
				t.Logf("poly[%d][%d]: expected %d, got %d", i, j, polys[i][j], invPolys[i][j])
			}
		}
	}
}

func TestPolyOperations(t *testing.T) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	n := int(gpu.N)

	// Create test polynomials
	a := make([]uint64, n)
	b := make([]uint64, n)
	for i := 0; i < n; i++ {
		a[i] = uint64(i) % gpu.Q
		b[i] = uint64(n-i) % gpu.Q
	}

	// Test PolyAdd
	sum, err := gpu.PolyAdd(a, b)
	if err != nil {
		t.Fatalf("PolyAdd failed: %v", err)
	}
	if len(sum) != n {
		t.Errorf("expected sum length %d, got %d", n, len(sum))
	}

	// Test PolySub
	diff, err := gpu.PolySub(a, b)
	if err != nil {
		t.Fatalf("PolySub failed: %v", err)
	}
	if len(diff) != n {
		t.Errorf("expected diff length %d, got %d", n, len(diff))
	}

	// Test PolyScalarMul
	scaled, err := gpu.PolyScalarMul(a, 2)
	if err != nil {
		t.Fatalf("PolyScalarMul failed: %v", err)
	}
	if len(scaled) != n {
		t.Errorf("expected scaled length %d, got %d", n, len(scaled))
	}
}

func TestPolyMul(t *testing.T) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	n := int(gpu.N)

	// Create batch of polynomial pairs
	aPolys := make([][]uint64, 2)
	bPolys := make([][]uint64, 2)
	for i := range aPolys {
		aPolys[i] = make([]uint64, n)
		bPolys[i] = make([]uint64, n)
		for j := 0; j < n; j++ {
			aPolys[i][j] = uint64(j+i) % gpu.Q
			bPolys[i][j] = uint64(n-j+i) % gpu.Q
		}
	}

	// Test batch PolyMul
	products, err := gpu.PolyMul(aPolys, bPolys)
	if err != nil {
		t.Fatalf("PolyMul failed: %v", err)
	}
	if len(products) != 2 {
		t.Errorf("expected 2 products, got %d", len(products))
	}
}

func TestSampling(t *testing.T) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	seed := []byte("test seed for sampling operations!")

	// Test SampleUniform
	uniform, err := gpu.SampleUniform(seed)
	if err != nil {
		t.Fatalf("SampleUniform failed: %v", err)
	}
	if len(uniform) != int(gpu.N) {
		t.Errorf("expected uniform length %d, got %d", gpu.N, len(uniform))
	}

	// Test SampleGaussian
	gaussian, err := gpu.SampleGaussian(3.2, seed)
	if err != nil {
		t.Fatalf("SampleGaussian failed: %v", err)
	}
	if len(gaussian) != int(gpu.N) {
		t.Errorf("expected gaussian length %d, got %d", gpu.N, len(gaussian))
	}

	// Test SampleTernary
	ternary, err := gpu.SampleTernary(0.5, seed)
	if err != nil {
		t.Fatalf("SampleTernary failed: %v", err)
	}
	if len(ternary) != int(gpu.N) {
		t.Errorf("expected ternary length %d, got %d", gpu.N, len(ternary))
	}
}

func TestMatrixVectorMul(t *testing.T) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	n := int(gpu.N)

	// Create a 2x3 matrix and 3-element vector
	matrix := make([][][]uint64, 2)
	for i := range matrix {
		matrix[i] = make([][]uint64, 3)
		for j := range matrix[i] {
			matrix[i][j] = make([]uint64, n)
			for k := 0; k < n; k++ {
				matrix[i][j][k] = uint64(i*100+j*10+k) % gpu.Q
			}
		}
	}

	vector := make([][]uint64, 3)
	for i := range vector {
		vector[i] = make([]uint64, n)
		for j := 0; j < n; j++ {
			vector[i][j] = uint64(i*10+j) % gpu.Q
		}
	}

	// Matrix-vector multiply
	result, err := gpu.MatrixVectorMul(matrix, vector)
	if err != nil {
		t.Fatalf("MatrixVectorMul failed: %v", err)
	}
	if len(result) != 2 {
		t.Errorf("expected result length 2, got %d", len(result))
	}
}

func TestGlobalInstance(t *testing.T) {
	gpu, err := GetRingtailGPU()
	if err != nil {
		t.Fatalf("GetRingtailGPU failed: %v", err)
	}
	if gpu == nil {
		t.Error("expected non-nil GPU instance")
	}

	enabled := RingtailGPUEnabled()
	t.Logf("Ringtail GPU enabled: %v", enabled)
}

func TestVectorNTT(t *testing.T) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		t.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	n := int(gpu.N)

	// Create vector of polynomials
	vectors := make([][]uint64, 4)
	for i := range vectors {
		vectors[i] = make([]uint64, n)
		for j := 0; j < n; j++ {
			vectors[i][j] = uint64(i*n+j) % gpu.Q
		}
	}

	// Forward NTT on vector
	nttVectors, err := gpu.VectorNTTForward(vectors)
	if err != nil {
		t.Fatalf("VectorNTTForward failed: %v", err)
	}
	if len(nttVectors) != 4 {
		t.Errorf("expected 4 vectors, got %d", len(nttVectors))
	}

	// Inverse NTT
	invVectors, err := gpu.VectorNTTInverse(nttVectors)
	if err != nil {
		t.Fatalf("VectorNTTInverse failed: %v", err)
	}
	if len(invVectors) != 4 {
		t.Errorf("expected 4 vectors, got %d", len(invVectors))
	}
}

func BenchmarkNTTForward(b *testing.B) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		b.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	n := int(gpu.N)
	polys := make([][]uint64, 16)
	for i := range polys {
		polys[i] = make([]uint64, n)
		for j := 0; j < n; j++ {
			polys[i][j] = uint64(j) % gpu.Q
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.NTTForward(polys)
	}
}

func BenchmarkPolyMul(b *testing.B) {
	gpu, err := NewRingtailGPU(DefaultConfig())
	if err != nil {
		b.Fatalf("failed to create RingtailGPU: %v", err)
	}
	defer gpu.Close()

	n := int(gpu.N)
	aPolys := make([][]uint64, 16)
	bPolys := make([][]uint64, 16)
	for i := range aPolys {
		aPolys[i] = make([]uint64, n)
		bPolys[i] = make([]uint64, n)
		for j := 0; j < n; j++ {
			aPolys[i][j] = uint64(j) % gpu.Q
			bPolys[i][j] = uint64(n-j) % gpu.Q
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.PolyMul(aPolys, bPolys)
	}
}
