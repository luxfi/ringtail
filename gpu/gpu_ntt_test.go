package gpu

import (
	"testing"
)

func TestNewBatchNTT(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	if ntt.N != 256 {
		t.Errorf("expected N=256, got %d", ntt.N)
	}
	if ntt.Q != 0x1000000004A01 {
		t.Errorf("expected Q=0x1000000004A01, got %x", ntt.Q)
	}
	if !ntt.initialized {
		t.Error("expected initialized=true")
	}
}

func TestNTTRoundTrip(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	// Create test polynomial
	poly := make([]uint64, ntt.N)
	for i := range poly {
		poly[i] = uint64(i * 17 % int(ntt.Q))
	}

	// Forward NTT
	nttPoly := ntt.ForwardSingle(poly)
	if nttPoly == nil {
		t.Fatal("ForwardSingle returned nil")
	}

	// Inverse NTT
	recovered := ntt.InverseSingle(nttPoly)
	if recovered == nil {
		t.Fatal("InverseSingle returned nil")
	}

	// Verify round-trip
	for i := range poly {
		if recovered[i] != poly[i] {
			t.Errorf("round-trip failed at index %d: expected %d, got %d", i, poly[i], recovered[i])
		}
	}
}

func TestBatchNTTRoundTrip(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	// Create test polynomials
	polys := make([][]uint64, 8)
	for i := range polys {
		polys[i] = make([]uint64, ntt.N)
		for j := range polys[i] {
			polys[i][j] = uint64((i*256 + j) % int(ntt.Q))
		}
	}

	// Forward NTT
	nttPolys := ntt.Forward(polys)
	if nttPolys == nil {
		t.Fatal("Forward returned nil")
	}

	// Inverse NTT
	recovered := ntt.Inverse(nttPolys)
	if recovered == nil {
		t.Fatal("Inverse returned nil")
	}

	// Verify round-trip
	for i := range polys {
		for j := range polys[i] {
			if recovered[i][j] != polys[i][j] {
				t.Errorf("batch round-trip failed at [%d][%d]: expected %d, got %d",
					i, j, polys[i][j], recovered[i][j])
			}
		}
	}
}

func TestNTTMultiplication(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	// NTT computes negacyclic convolution in R_q = Z_q[x]/(x^N + 1)
	// For polynomial multiplication:
	// (1 + x) * (1 + x) = 1 + 2x + x^2
	// But in negacyclic ring, we need to test with smaller polynomials

	// Simple test: multiply by 1 (constant polynomial)
	a := make([]uint64, ntt.N)
	b := make([]uint64, ntt.N)
	a[0] = 5
	a[1] = 3
	b[0] = 1 // Multiply by 1

	// Convert to NTT domain
	aNTT := ntt.ForwardSingle(a)
	bNTT := ntt.ForwardSingle(b)

	// Multiply in NTT domain (Hadamard product)
	cNTT := make([]uint64, ntt.N)
	for i := range cNTT {
		cNTT[i] = mulMod(aNTT[i], bNTT[i], ntt.Q)
	}

	// Convert back
	c := ntt.InverseSingle(cNTT)

	// Multiplying by 1 should give back the original
	if c[0] != 5 {
		t.Errorf("c[0] expected 5, got %d", c[0])
	}
	if c[1] != 3 {
		t.Errorf("c[1] expected 3, got %d", c[1])
	}

	// Test multiplication by scalar 2
	b2 := make([]uint64, ntt.N)
	b2[0] = 2

	b2NTT := ntt.ForwardSingle(b2)
	c2NTT := make([]uint64, ntt.N)
	for i := range c2NTT {
		c2NTT[i] = mulMod(aNTT[i], b2NTT[i], ntt.Q)
	}
	c2 := ntt.InverseSingle(c2NTT)

	if c2[0] != 10 {
		t.Errorf("c2[0] expected 10, got %d", c2[0])
	}
	if c2[1] != 6 {
		t.Errorf("c2[1] expected 6, got %d", c2[1])
	}
}

func TestMontgomeryArithmetic(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	// Test Montgomery conversion round-trip
	testVals := []uint64{0, 1, 2, 100, 1000, ntt.Q - 1, ntt.Q / 2}
	for _, v := range testVals {
		mont := ntt.toMontgomery(v)
		back := ntt.fromMontgomery(mont)
		if back != v {
			t.Errorf("Montgomery round-trip failed for %d: got %d", v, back)
		}
	}

	// Test Montgomery multiplication
	a := uint64(12345)
	b := uint64(67890)
	expected := mulMod(a, b, ntt.Q)

	aMont := ntt.toMontgomery(a)
	bMont := ntt.toMontgomery(b)
	cMont := ntt.montMul(aMont, bMont)
	actual := ntt.fromMontgomery(cMont)

	if actual != expected {
		t.Errorf("Montgomery mul failed: %d * %d mod Q = %d, got %d", a, b, expected, actual)
	}
}

func TestModularArithmetic(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	// Test modAdd
	if got := ntt.modAdd(ntt.Q-1, 2); got != 1 {
		t.Errorf("modAdd(Q-1, 2) = %d, want 1", got)
	}

	// Test modSub
	if got := ntt.modSub(1, 2); got != ntt.Q-1 {
		t.Errorf("modSub(1, 2) = %d, want Q-1", got)
	}
}

func TestModInverse(t *testing.T) {
	Q := DefaultQ
	testVals := []uint64{1, 2, 7, 256, 12345}
	for _, v := range testVals {
		inv := modInverse(v, Q)
		prod := mulMod(v, inv, Q)
		if prod != 1 {
			t.Errorf("modInverse(%d) failed: %d * %d mod Q = %d, want 1", v, v, inv, prod)
		}
	}
}

func TestBitReverse(t *testing.T) {
	// Test for N=256, log2(N)=8
	tests := []struct {
		in, out uint32
	}{
		{0, 0},
		{1, 128},
		{2, 64},
		{128, 1},
		{255, 255},
	}

	for _, tc := range tests {
		got := bitReverse(tc.in, 8)
		if got != tc.out {
			t.Errorf("bitReverse(%d, 8) = %d, want %d", tc.in, got, tc.out)
		}
	}
}

func BenchmarkBatchNTTForward(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	poly := make([]uint64, ntt.N)
	for i := range poly {
		poly[i] = uint64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ntt.ForwardSingle(poly)
	}
}

func BenchmarkBatchNTTInverse(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	poly := make([]uint64, ntt.N)
	for i := range poly {
		poly[i] = uint64(i)
	}
	nttPoly := ntt.ForwardSingle(poly)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ntt.InverseSingle(nttPoly)
	}
}

func BenchmarkBatchNTT8(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	polys := make([][]uint64, 8)
	for i := range polys {
		polys[i] = make([]uint64, ntt.N)
		for j := range polys[i] {
			polys[i][j] = uint64(i*256 + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ntt.Forward(polys)
	}
}
