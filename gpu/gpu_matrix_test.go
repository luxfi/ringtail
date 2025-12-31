package gpu

import (
	"testing"
)

func TestNewGPUMatrix(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	m := NewGPUMatrix(4, 3, ntt)
	if m == nil {
		t.Fatal("NewGPUMatrix returned nil")
	}

	if m.Rows() != 4 {
		t.Errorf("expected rows=4, got %d", m.Rows())
	}
	if m.Cols() != 3 {
		t.Errorf("expected cols=3, got %d", m.Cols())
	}
	if m.IsNTT() {
		t.Error("expected IsNTT=false for new matrix")
	}
}

func TestGPUMatrixSetGet(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	m := NewGPUMatrix(2, 2, ntt)

	// Create test polynomial
	poly := make([]uint64, ntt.N)
	for i := range poly {
		poly[i] = uint64(i + 1)
	}

	m.Set(0, 1, poly)
	got := m.Get(0, 1)

	for i := range poly {
		if got[i] != poly[i] {
			t.Errorf("Get[0][1][%d] = %d, want %d", i, got[i], poly[i])
		}
	}
}

func TestGPUMatrixToFromNTT(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	m := NewGPUMatrix(2, 2, ntt)

	// Set some values
	for row := uint32(0); row < 2; row++ {
		for col := uint32(0); col < 2; col++ {
			poly := make([]uint64, ntt.N)
			for i := range poly {
				poly[i] = uint64(row*100 + col*10 + uint32(i)%10)
			}
			m.Set(row, col, poly)
		}
	}

	// Convert to NTT
	mNTT := m.ToNTT()
	if !mNTT.IsNTT() {
		t.Error("expected IsNTT=true after ToNTT")
	}

	// Convert back
	mCoeff := mNTT.FromNTT()
	if mCoeff.IsNTT() {
		t.Error("expected IsNTT=false after FromNTT")
	}

	// Verify round-trip
	for row := uint32(0); row < 2; row++ {
		for col := uint32(0); col < 2; col++ {
			orig := m.Get(row, col)
			got := mCoeff.Get(row, col)
			for i := range orig {
				if got[i] != orig[i] {
					t.Errorf("round-trip failed at [%d][%d][%d]: expected %d, got %d",
						row, col, i, orig[i], got[i])
				}
			}
		}
	}
}

func TestGPUMatrixMulVecPoly(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	// Create identity-like matrix (each element is constant 1)
	m := NewGPUMatrix(2, 2, ntt)
	one := make([]uint64, ntt.N)
	one[0] = 1
	m.Set(0, 0, one)
	m.Set(1, 1, one)

	// Convert to NTT
	mNTT := m.ToNTT()

	// Create vector
	v := make([][]uint64, 2)
	for i := range v {
		v[i] = make([]uint64, ntt.N)
		v[i][0] = uint64(i + 1) // [1, 2]
	}

	// Convert vector to NTT
	vNTT := ntt.Forward(v)

	// Multiply
	result := mNTT.MulVecPoly(vNTT)
	if result == nil {
		t.Fatal("MulVecPoly returned nil")
	}

	// Convert back
	resultCoeff := ntt.Inverse(result)

	// For diagonal matrix with 1s: result should be same as input
	if resultCoeff[0][0] != 1 {
		t.Errorf("result[0][0] = %d, want 1", resultCoeff[0][0])
	}
	if resultCoeff[1][0] != 2 {
		t.Errorf("result[1][0] = %d, want 2", resultCoeff[1][0])
	}
}

func TestGPUMatrixAdd(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	m1 := NewGPUMatrix(2, 2, ntt)
	m2 := NewGPUMatrix(2, 2, ntt)

	// Set values
	poly1 := make([]uint64, ntt.N)
	poly2 := make([]uint64, ntt.N)
	for i := range poly1 {
		poly1[i] = 5
		poly2[i] = 3
	}

	m1.Set(0, 0, poly1)
	m2.Set(0, 0, poly2)

	result := m1.Add(m2)
	if result == nil {
		t.Fatal("Add returned nil")
	}

	got := result.Get(0, 0)
	if got[0] != 8 {
		t.Errorf("Add result[0][0][0] = %d, want 8", got[0])
	}
}

func TestGPUMatrixSub(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	m1 := NewGPUMatrix(2, 2, ntt)
	m2 := NewGPUMatrix(2, 2, ntt)

	poly1 := make([]uint64, ntt.N)
	poly2 := make([]uint64, ntt.N)
	for i := range poly1 {
		poly1[i] = 10
		poly2[i] = 3
	}

	m1.Set(0, 0, poly1)
	m2.Set(0, 0, poly2)

	result := m1.Sub(m2)
	if result == nil {
		t.Fatal("Sub returned nil")
	}

	got := result.Get(0, 0)
	if got[0] != 7 {
		t.Errorf("Sub result[0][0][0] = %d, want 7", got[0])
	}
}

func TestGPUMatrixClone(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	m := NewGPUMatrix(2, 2, ntt)
	poly := make([]uint64, ntt.N)
	poly[0] = 42
	m.Set(0, 0, poly)

	clone := m.Clone()

	// Modify original
	poly2 := make([]uint64, ntt.N)
	poly2[0] = 99
	m.Set(0, 0, poly2)

	// Clone should be unchanged
	got := clone.Get(0, 0)
	if got[0] != 42 {
		t.Errorf("Clone was modified: got %d, want 42", got[0])
	}
}

func TestGPUMatrixTranspose(t *testing.T) {
	ntt, err := NewBatchNTT(DefaultN, DefaultQ)
	if err != nil {
		t.Fatalf("NewBatchNTT failed: %v", err)
	}
	defer ntt.Close()

	m := NewGPUMatrix(2, 3, ntt)

	// Set some values
	for row := uint32(0); row < 2; row++ {
		for col := uint32(0); col < 3; col++ {
			poly := make([]uint64, ntt.N)
			poly[0] = uint64(row*10 + col)
			m.Set(row, col, poly)
		}
	}

	mt := m.Transpose()
	if mt.Rows() != 3 || mt.Cols() != 2 {
		t.Errorf("Transpose dimensions: got %dx%d, want 3x2", mt.Rows(), mt.Cols())
	}

	// Check values
	for row := uint32(0); row < 2; row++ {
		for col := uint32(0); col < 3; col++ {
			orig := m.Get(row, col)
			trans := mt.Get(col, row)
			if orig[0] != trans[0] {
				t.Errorf("Transpose[%d][%d] = %d, want %d", col, row, trans[0], orig[0])
			}
		}
	}
}

func BenchmarkGPUMatrixToNTT(b *testing.B) {
	ntt, _ := NewBatchNTT(DefaultN, DefaultQ)
	defer ntt.Close()

	m := NewGPUMatrix(8, 7, ntt) // Typical Ringtail dimensions
	for row := uint32(0); row < 8; row++ {
		for col := uint32(0); col < 7; col++ {
			poly := make([]uint64, ntt.N)
			for i := range poly {
				poly[i] = uint64(i)
			}
			m.Set(row, col, poly)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.ToNTT()
	}
}
