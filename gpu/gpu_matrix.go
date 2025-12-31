// Package gpu provides GPU-accelerated matrix operations for Ringtail threshold signatures.
// This file implements matrix operations using NTT-domain multiplication.
package gpu

import (
	"sync"
)

// GPUMatrix represents a matrix of polynomials for GPU-accelerated operations.
// Polynomials can be stored in either coefficient or NTT domain.
type GPUMatrix struct {
	mu sync.RWMutex

	rows uint32      // Number of rows
	cols uint32      // Number of columns
	data [][]uint64  // Matrix data: data[row*cols + col] is polynomial
	ntt  *BatchNTT   // NTT context for transformations

	isNTT bool // True if data is in NTT domain
}

// NewGPUMatrix creates a new matrix of polynomials.
// Matrix elements are initialized to zero polynomials.
func NewGPUMatrix(rows, cols uint32, ntt *BatchNTT) *GPUMatrix {
	if ntt == nil {
		return nil
	}

	m := &GPUMatrix{
		rows:  rows,
		cols:  cols,
		data:  make([][]uint64, rows*cols),
		ntt:   ntt,
		isNTT: false,
	}

	// Initialize all polynomials to zero
	for i := range m.data {
		m.data[i] = make([]uint64, ntt.N)
	}

	return m
}

// NewGPUMatrixFromData creates a matrix from existing polynomial data.
// Data should be provided in row-major order.
func NewGPUMatrixFromData(rows, cols uint32, data [][]uint64, ntt *BatchNTT, isNTT bool) *GPUMatrix {
	if ntt == nil || uint32(len(data)) != rows*cols {
		return nil
	}

	// Deep copy the data
	dataCopy := make([][]uint64, len(data))
	for i := range data {
		dataCopy[i] = make([]uint64, len(data[i]))
		copy(dataCopy[i], data[i])
	}

	return &GPUMatrix{
		rows:  rows,
		cols:  cols,
		data:  dataCopy,
		ntt:   ntt,
		isNTT: isNTT,
	}
}

// Rows returns the number of rows
func (m *GPUMatrix) Rows() uint32 {
	return m.rows
}

// Cols returns the number of columns
func (m *GPUMatrix) Cols() uint32 {
	return m.cols
}

// Get returns the polynomial at position (row, col)
func (m *GPUMatrix) Get(row, col uint32) []uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if row >= m.rows || col >= m.cols {
		return nil
	}
	return m.data[row*m.cols+col]
}

// Set sets the polynomial at position (row, col)
func (m *GPUMatrix) Set(row, col uint32, poly []uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if row >= m.rows || col >= m.cols {
		return
	}
	idx := row*m.cols + col
	if m.data[idx] == nil {
		m.data[idx] = make([]uint64, len(poly))
	}
	copy(m.data[idx], poly)
}

// IsNTT returns true if the matrix is in NTT domain
func (m *GPUMatrix) IsNTT() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.isNTT
}

// Data returns the underlying data slice (for advanced use)
func (m *GPUMatrix) Data() [][]uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.data
}

// Clone creates a deep copy of the matrix
func (m *GPUMatrix) Clone() *GPUMatrix {
	m.mu.RLock()
	defer m.mu.RUnlock()

	dataCopy := make([][]uint64, len(m.data))
	for i := range m.data {
		dataCopy[i] = make([]uint64, len(m.data[i]))
		copy(dataCopy[i], m.data[i])
	}

	return &GPUMatrix{
		rows:  m.rows,
		cols:  m.cols,
		data:  dataCopy,
		ntt:   m.ntt,
		isNTT: m.isNTT,
	}
}

// =============================================================================
// Domain Conversion
// =============================================================================

// ToNTT converts the matrix to NTT domain.
// Returns a new matrix; original is unchanged.
func (m *GPUMatrix) ToNTT() *GPUMatrix {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.isNTT {
		return m.Clone()
	}

	// Convert all polynomials to NTT domain in parallel
	nttData := m.ntt.Forward(m.data)

	result := &GPUMatrix{
		rows:  m.rows,
		cols:  m.cols,
		data:  nttData,
		ntt:   m.ntt,
		isNTT: true,
	}

	return result
}

// FromNTT converts the matrix from NTT domain to coefficient domain.
// Returns a new matrix; original is unchanged.
func (m *GPUMatrix) FromNTT() *GPUMatrix {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.isNTT {
		return m.Clone()
	}

	// Convert all polynomials from NTT domain in parallel
	coeffData := m.ntt.Inverse(m.data)

	result := &GPUMatrix{
		rows:  m.rows,
		cols:  m.cols,
		data:  coeffData,
		ntt:   m.ntt,
		isNTT: false,
	}

	return result
}

// =============================================================================
// Matrix-Vector Operations
// =============================================================================

// MulVec computes matrix-vector multiplication: result = M * v
// Both matrix and vector should be in NTT domain for efficiency.
// Vector v should have length equal to matrix columns.
// Returns a vector of length equal to matrix rows.
func (m *GPUMatrix) MulVec(v []uint64) []uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.cols == 0 {
		return nil
	}

	// For polynomial vectors, each element is a polynomial
	// v is a flattened vector of cols polynomials
	polyLen := uint32(len(v)) / m.cols
	if polyLen != m.ntt.N {
		return nil
	}

	result := make([]uint64, m.rows*m.ntt.N)

	// Compute each row
	var wg sync.WaitGroup
	for i := uint32(0); i < m.rows; i++ {
		wg.Add(1)
		go func(row uint32) {
			defer wg.Done()
			rowResult := make([]uint64, m.ntt.N)

			for j := uint32(0); j < m.cols; j++ {
				// Get matrix element M[row][col]
				matPoly := m.data[row*m.cols+j]

				// Get vector element v[col]
				vecPoly := v[j*m.ntt.N : (j+1)*m.ntt.N]

				// Multiply (Hadamard product in NTT domain)
				prod := hadamardProduct(matPoly, vecPoly, m.ntt.Q)

				// Add to result
				polyAdd(rowResult, prod, m.ntt.Q)
			}

			// Store result
			copy(result[row*m.ntt.N:(row+1)*m.ntt.N], rowResult)
		}(i)
	}
	wg.Wait()

	return result
}

// MulVecPoly computes matrix-vector multiplication where vector is [][]uint64.
// Each element of vector is a polynomial.
func (m *GPUMatrix) MulVecPoly(v [][]uint64) [][]uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if uint32(len(v)) != m.cols {
		return nil
	}

	result := make([][]uint64, m.rows)

	var wg sync.WaitGroup
	for i := uint32(0); i < m.rows; i++ {
		wg.Add(1)
		go func(row uint32) {
			defer wg.Done()
			rowResult := make([]uint64, m.ntt.N)

			for j := uint32(0); j < m.cols; j++ {
				matPoly := m.data[row*m.cols+j]
				vecPoly := v[j]

				// Hadamard product in NTT domain
				prod := hadamardProduct(matPoly, vecPoly, m.ntt.Q)

				// Accumulate
				polyAdd(rowResult, prod, m.ntt.Q)
			}

			result[row] = rowResult
		}(i)
	}
	wg.Wait()

	return result
}

// =============================================================================
// Matrix-Matrix Operations
// =============================================================================

// MulMat computes matrix-matrix multiplication: result = M * other
// Both matrices should be in NTT domain.
// Dimensions must be compatible: M[rows x cols] * other[cols x other.cols]
func (m *GPUMatrix) MulMat(other *GPUMatrix) *GPUMatrix {
	m.mu.RLock()
	other.mu.RLock()
	defer m.mu.RUnlock()
	defer other.mu.RUnlock()

	if m.cols != other.rows || m.ntt != other.ntt {
		return nil
	}

	result := NewGPUMatrix(m.rows, other.cols, m.ntt)
	result.isNTT = true

	// Compute C[i][j] = sum_k(A[i][k] * B[k][j])
	var wg sync.WaitGroup
	for i := uint32(0); i < m.rows; i++ {
		for j := uint32(0); j < other.cols; j++ {
			wg.Add(1)
			go func(row, col uint32) {
				defer wg.Done()
				sum := make([]uint64, m.ntt.N)

				for k := uint32(0); k < m.cols; k++ {
					aPoly := m.data[row*m.cols+k]
					bPoly := other.data[k*other.cols+col]

					prod := hadamardProduct(aPoly, bPoly, m.ntt.Q)
					polyAdd(sum, prod, m.ntt.Q)
				}

				result.data[row*other.cols+col] = sum
			}(i, j)
		}
	}
	wg.Wait()

	return result
}

// Hadamard computes element-wise (Hadamard) product of two matrices.
// Both matrices must have the same dimensions and be in NTT domain.
func (m *GPUMatrix) Hadamard(other *GPUMatrix) *GPUMatrix {
	m.mu.RLock()
	other.mu.RLock()
	defer m.mu.RUnlock()
	defer other.mu.RUnlock()

	if m.rows != other.rows || m.cols != other.cols || m.ntt != other.ntt {
		return nil
	}

	result := NewGPUMatrix(m.rows, m.cols, m.ntt)
	result.isNTT = m.isNTT && other.isNTT

	var wg sync.WaitGroup
	for i := uint32(0); i < m.rows*m.cols; i++ {
		wg.Add(1)
		go func(idx uint32) {
			defer wg.Done()
			result.data[idx] = hadamardProduct(m.data[idx], other.data[idx], m.ntt.Q)
		}(i)
	}
	wg.Wait()

	return result
}

// Add computes matrix addition: result = M + other
func (m *GPUMatrix) Add(other *GPUMatrix) *GPUMatrix {
	m.mu.RLock()
	other.mu.RLock()
	defer m.mu.RUnlock()
	defer other.mu.RUnlock()

	if m.rows != other.rows || m.cols != other.cols || m.ntt != other.ntt {
		return nil
	}

	result := NewGPUMatrix(m.rows, m.cols, m.ntt)
	result.isNTT = m.isNTT

	var wg sync.WaitGroup
	for i := uint32(0); i < m.rows*m.cols; i++ {
		wg.Add(1)
		go func(idx uint32) {
			defer wg.Done()
			result.data[idx] = polyAddNew(m.data[idx], other.data[idx], m.ntt.Q)
		}(i)
	}
	wg.Wait()

	return result
}

// Sub computes matrix subtraction: result = M - other
func (m *GPUMatrix) Sub(other *GPUMatrix) *GPUMatrix {
	m.mu.RLock()
	other.mu.RLock()
	defer m.mu.RUnlock()
	defer other.mu.RUnlock()

	if m.rows != other.rows || m.cols != other.cols || m.ntt != other.ntt {
		return nil
	}

	result := NewGPUMatrix(m.rows, m.cols, m.ntt)
	result.isNTT = m.isNTT

	var wg sync.WaitGroup
	for i := uint32(0); i < m.rows*m.cols; i++ {
		wg.Add(1)
		go func(idx uint32) {
			defer wg.Done()
			result.data[idx] = polySubNew(m.data[idx], other.data[idx], m.ntt.Q)
		}(i)
	}
	wg.Wait()

	return result
}

// ScalarMul multiplies all elements by a scalar polynomial
func (m *GPUMatrix) ScalarMul(scalar []uint64) *GPUMatrix {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := NewGPUMatrix(m.rows, m.cols, m.ntt)
	result.isNTT = m.isNTT

	var wg sync.WaitGroup
	for i := uint32(0); i < m.rows*m.cols; i++ {
		wg.Add(1)
		go func(idx uint32) {
			defer wg.Done()
			result.data[idx] = hadamardProduct(m.data[idx], scalar, m.ntt.Q)
		}(i)
	}
	wg.Wait()

	return result
}

// =============================================================================
// Helper Functions
// =============================================================================

// hadamardProduct computes element-wise product of two polynomials
func hadamardProduct(a, b []uint64, Q uint64) []uint64 {
	n := len(a)
	if len(b) != n {
		return nil
	}

	result := make([]uint64, n)
	for i := 0; i < n; i++ {
		result[i] = mulMod(a[i], b[i], Q)
	}
	return result
}

// polyAdd adds polynomial b to a in-place: a += b
func polyAdd(a, b []uint64, Q uint64) {
	for i := range a {
		sum := a[i] + b[i]
		if sum >= Q {
			sum -= Q
		}
		a[i] = sum
	}
}

// polyAddNew creates a new polynomial: result = a + b
func polyAddNew(a, b []uint64, Q uint64) []uint64 {
	result := make([]uint64, len(a))
	for i := range a {
		sum := a[i] + b[i]
		if sum >= Q {
			sum -= Q
		}
		result[i] = sum
	}
	return result
}

// polySubNew creates a new polynomial: result = a - b
func polySubNew(a, b []uint64, Q uint64) []uint64 {
	result := make([]uint64, len(a))
	for i := range a {
		if a[i] >= b[i] {
			result[i] = a[i] - b[i]
		} else {
			result[i] = Q - b[i] + a[i]
		}
	}
	return result
}

// Transpose returns the transpose of the matrix
func (m *GPUMatrix) Transpose() *GPUMatrix {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := NewGPUMatrix(m.cols, m.rows, m.ntt)
	result.isNTT = m.isNTT

	for i := uint32(0); i < m.rows; i++ {
		for j := uint32(0); j < m.cols; j++ {
			srcIdx := i*m.cols + j
			dstIdx := j*m.rows + i
			copy(result.data[dstIdx], m.data[srcIdx])
		}
	}

	return result
}
