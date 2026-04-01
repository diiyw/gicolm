//go:build !amd64

package main

import (
	"encoding/binary"
	"math"
)

// VecDotF32F32 computes the dot product of two float32 vectors.
// Unrolled 4x for SIMD auto-vectorization.
func VecDotF32F32(src []byte, x []float32, n int) float32 {
	var sum float32
	i := 0
	for i+4 <= n {
		w0 := math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		w1 := math.Float32frombits(binary.LittleEndian.Uint32(src[(i+1)*4:]))
		w2 := math.Float32frombits(binary.LittleEndian.Uint32(src[(i+2)*4:]))
		w3 := math.Float32frombits(binary.LittleEndian.Uint32(src[(i+3)*4:]))
		sum += w0*x[i] + w1*x[i+1] + w2*x[i+2] + w3*x[i+3]
		i += 4
	}
	for i < n {
		w := math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		sum += w * x[i]
		i++
	}
	return sum
}

// ElemwiseMul computes out[i] = a[i] * b[i]
// Unrolled 4x for SIMD auto-vectorization.
func ElemwiseMul(out, a, b []float32, size int) {
	i := 0
	for i+4 <= size {
		out[i] = a[i] * b[i]
		out[i+1] = a[i+1] * b[i+1]
		out[i+2] = a[i+2] * b[i+2]
		out[i+3] = a[i+3] * b[i+3]
		i += 4
	}
	for i < size {
		out[i] = a[i] * b[i]
		i++
	}
}

// VecAdd computes a[i] += b[i] in-place.
// Unrolled 4x for SIMD auto-vectorization.
func VecAdd(a, b []float32, size int) {
	i := 0
	for i+4 <= size {
		a[i] += b[i]
		a[i+1] += b[i+1]
		a[i+2] += b[i+2]
		a[i+3] += b[i+3]
		i += 4
	}
	for i < size {
		a[i] += b[i]
		i++
	}
}

// RmsNorm computes out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
// Unrolled 4x for SIMD auto-vectorization.
func RmsNorm(out, x, weight []float32, size int) {
	var ss float32
	i := 0
	for i+4 <= size {
		ss += x[i]*x[i] + x[i+1]*x[i+1] + x[i+2]*x[i+2] + x[i+3]*x[i+3]
		i += 4
	}
	for i < size {
		ss += x[i] * x[i]
		i++
	}
	ss = 1.0 / float32(math.Sqrt(float64(ss/float32(size)+1e-5)))
	i = 0
	for i+4 <= size {
		out[i] = x[i] * ss * weight[i]
		out[i+1] = x[i+1] * ss * weight[i+1]
		out[i+2] = x[i+2] * ss * weight[i+2]
		out[i+3] = x[i+3] * ss * weight[i+3]
		i += 4
	}
	for i < size {
		out[i] = x[i] * ss * weight[i]
		i++
	}
}

// SiLU applies in-place SiLU activation: x[i] = x[i] / (1 + exp(-x[i]))
// Unrolled 4x for SIMD auto-vectorization.
func SiLU(x []float32, size int) {
	i := 0
	for i+4 <= size {
		x[i] = x[i] / (1.0 + float32(math.Exp(float64(-x[i]))))
		x[i+1] = x[i+1] / (1.0 + float32(math.Exp(float64(-x[i+1]))))
		x[i+2] = x[i+2] / (1.0 + float32(math.Exp(float64(-x[i+2]))))
		x[i+3] = x[i+3] / (1.0 + float32(math.Exp(float64(-x[i+3]))))
		i += 4
	}
	for i < size {
		x[i] = x[i] / (1.0 + float32(math.Exp(float64(-x[i]))))
		i++
	}
}
