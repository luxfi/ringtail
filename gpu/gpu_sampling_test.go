package gpu

import (
	"testing"
)

func TestSampleUniform(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	polys := ntt.SampleUniform(10)
	if len(polys) != 10 {
		t.Fatalf("expected 10 polynomials, got %d", len(polys))
	}

	// Check all coefficients are in range [0, Q-1]
	for i, poly := range polys {
		if len(poly) != int(ntt.N) {
			t.Errorf("poly[%d] has wrong length: %d", i, len(poly))
		}
		for j, coeff := range poly {
			if coeff >= ntt.Q {
				t.Errorf("poly[%d][%d] = %d >= Q", i, j, coeff)
			}
		}
	}

	// Check that polynomials are different (probabilistic)
	if polys[0][0] == polys[1][0] && polys[0][1] == polys[1][1] {
		t.Log("Warning: first two polynomials have same first two coefficients (unlikely)")
	}
}

func TestSampleGaussian(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	sigma := 6.0
	polys := ntt.SampleGaussian(10, sigma)
	if len(polys) != 10 {
		t.Fatalf("expected 10 polynomials, got %d", len(polys))
	}

	// Check all coefficients are in valid range
	// Gaussian samples should be centered near 0 (or Q for negative)
	for i, poly := range polys {
		if len(poly) != int(ntt.N) {
			t.Errorf("poly[%d] has wrong length: %d", i, len(poly))
		}
		for j, coeff := range poly {
			if coeff >= ntt.Q {
				t.Errorf("poly[%d][%d] = %d >= Q", i, j, coeff)
			}
		}
	}
}

func TestSampleTernary(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	polys := ntt.SampleTernary(10)
	if len(polys) != 10 {
		t.Fatalf("expected 10 polynomials, got %d", len(polys))
	}

	// Check all coefficients are in {0, 1, Q-1}
	for i, poly := range polys {
		if len(poly) != int(ntt.N) {
			t.Errorf("poly[%d] has wrong length: %d", i, len(poly))
		}
		for j, coeff := range poly {
			if coeff != 0 && coeff != 1 && coeff != ntt.Q-1 {
				t.Errorf("poly[%d][%d] = %d, not in {0, 1, Q-1}", i, j, coeff)
			}
		}
	}

	// Check distribution is roughly balanced
	counts := make(map[uint64]int)
	for _, poly := range polys {
		for _, coeff := range poly {
			counts[coeff]++
		}
	}

	total := 10 * int(ntt.N)
	// Each value should be roughly 1/3 of total
	for val, count := range counts {
		ratio := float64(count) / float64(total)
		if ratio < 0.2 || ratio > 0.5 {
			t.Logf("Warning: ternary value %d has unusual frequency: %.2f", val, ratio)
		}
	}
}

func TestSampleUniformSeeded(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	seed := []byte("test seed 12345")

	// Same seed should produce same output
	polys1 := ntt.SampleUniformSeeded(5, seed)
	polys2 := ntt.SampleUniformSeeded(5, seed)

	for i := range polys1 {
		for j := range polys1[i] {
			if polys1[i][j] != polys2[i][j] {
				t.Errorf("seeded samples differ at [%d][%d]: %d vs %d",
					i, j, polys1[i][j], polys2[i][j])
			}
		}
	}

	// Different seed should produce different output
	seed2 := []byte("different seed")
	polys3 := ntt.SampleUniformSeeded(5, seed2)

	same := true
	for i := range polys1 {
		for j := range polys1[i] {
			if polys1[i][j] != polys3[i][j] {
				same = false
				break
			}
		}
		if !same {
			break
		}
	}
	if same {
		t.Error("different seeds produced same output")
	}
}

func TestSampleTernarySeeded(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	seed := []byte("ternary seed")

	polys1 := ntt.SampleTernarySeeded(5, seed)
	polys2 := ntt.SampleTernarySeeded(5, seed)

	for i := range polys1 {
		for j := range polys1[i] {
			if polys1[i][j] != polys2[i][j] {
				t.Errorf("seeded ternary samples differ at [%d][%d]", i, j)
			}
		}
	}
}

func TestSampleSecretKey(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	eta := 3
	sk := ntt.SampleSecretKey(eta)
	if len(sk) != int(ntt.N) {
		t.Fatalf("secret key has wrong length: %d", len(sk))
	}

	// Coefficients should be small (in [-eta, eta])
	for i, coeff := range sk {
		// Check if coefficient is small positive or negative (Q - small)
		isSmall := coeff <= uint64(eta) || coeff >= ntt.Q-uint64(eta)
		if !isSmall {
			t.Errorf("sk[%d] = %d, not in [-eta, eta]", i, coeff)
		}
	}
}

func TestSampleMatrix(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	seed := []byte("matrix seed")
	m := ntt.SampleMatrix(4, 3, seed)
	if m == nil {
		t.Fatal("SampleMatrix returned nil")
	}

	if m.Rows() != 4 || m.Cols() != 3 {
		t.Errorf("matrix dimensions: %dx%d, want 4x3", m.Rows(), m.Cols())
	}

	// Same seed should produce same matrix
	m2 := ntt.SampleMatrix(4, 3, seed)
	for row := uint32(0); row < 4; row++ {
		for col := uint32(0); col < 3; col++ {
			p1 := m.Get(row, col)
			p2 := m2.Get(row, col)
			for i := range p1 {
				if p1[i] != p2[i] {
					t.Errorf("seeded matrices differ at [%d][%d][%d]", row, col, i)
				}
			}
		}
	}
}

func TestExpandSeed(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	seed := []byte("expand me")
	polys := ntt.ExpandSeed(seed, 8)
	if len(polys) != 8 {
		t.Fatalf("expected 8 polynomials, got %d", len(polys))
	}

	// Verify deterministic
	polys2 := ntt.ExpandSeed(seed, 8)
	for i := range polys {
		for j := range polys[i] {
			if polys[i][j] != polys2[i][j] {
				t.Errorf("ExpandSeed not deterministic at [%d][%d]", i, j)
			}
		}
	}
}

func BenchmarkSampleUniform(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ntt.SampleUniform(8)
	}
}

func BenchmarkSampleGaussian(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ntt.SampleGaussian(8, 6.0)
	}
}

func BenchmarkSampleTernary(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ntt.SampleTernary(8)
	}
}

func BenchmarkSampleUniformSeeded(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	seed := []byte("benchmark seed")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ntt.SampleUniformSeeded(8, seed)
	}
}
