package main

import (
	"math"
	"runtime"
	"sync"
)

const maxThreads = 16

var nThreads = runtime.NumCPU()

func SetThreads(t int) {
	if t < 1 {
		t = 1
	}
	if t > maxThreads {
		t = maxThreads
	}
	nThreads = t
}

// MatMul performs out[d] = W[d, n] @ x[n] with quantized W.
func MatMul(out, x []float32, W []byte, n, d int, qtype GGUFType) {
	rowBytes := GGUFTypeRowSize(qtype, n)

	if nThreads <= 1 || d < 4 {
		for i := 0; i < d; i++ {
			wRow := W[i*rowBytes:]
			out[i] = VecDot(wRow, x, n, qtype)
		}
		return
	}

	nt := nThreads
	if nt > d {
		nt = d
	}

	var wg sync.WaitGroup
	rowsPer := d / nt
	extra := d % nt
	row := 0

	for t := 0; t < nt; t++ {
		start := row
		end := row + rowsPer
		if t < extra {
			end++
		}
		row = end

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				wRow := W[i*rowBytes:]
				out[i] = VecDot(wRow, x, n, qtype)
			}
		}(start, end)
	}

	wg.Wait()
}

// RmsNorm computes out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
func RmsNorm(out, x, weight []float32, size int) {
	var ss float32
	for i := 0; i < size; i++ {
		ss += x[i] * x[i]
	}
	ss = 1.0 / float32(math.Sqrt(float64(ss/float32(size)+1e-5)))
	for i := 0; i < size; i++ {
		out[i] = x[i] * ss * weight[i]
	}
}

// Softmax computes in-place softmax over x[0..size-1].
func Softmax(x []float32, size int) {
	maxVal := x[0]
	for i := 1; i < size; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := 0; i < size; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	inv := 1.0 / sum
	for i := 0; i < size; i++ {
		x[i] *= inv
	}
}

// RoPE applies rotary position encoding using pre-computed cos/sin tables.
func RoPE(q, k []float32, headDim, nHeads, nKVHeads int, cosPos, sinPos []float32) {
	half := headDim / 2

	for h := 0; h < nHeads; h++ {
		qh := q[h*headDim:]
		for i := 0; i < half; i++ {
			q0 := qh[i*2]
			q1 := qh[i*2+1]
			qh[i*2] = q0*cosPos[i] - q1*sinPos[i]
			qh[i*2+1] = q0*sinPos[i] + q1*cosPos[i]
		}
	}

	for h := 0; h < nKVHeads; h++ {
		kh := k[h*headDim:]
		for i := 0; i < half; i++ {
			k0 := kh[i*2]
			k1 := kh[i*2+1]
			kh[i*2] = k0*cosPos[i] - k1*sinPos[i]
			kh[i*2+1] = k0*sinPos[i] + k1*cosPos[i]
		}
	}
}

// SiLU applies in-place SiLU activation: x[i] = x[i] / (1 + exp(-x[i]))
func SiLU(x []float32, size int) {
	for i := 0; i < size; i++ {
		x[i] = x[i] / (1.0 + float32(math.Exp(float64(-x[i]))))
	}
}

// ElemwiseMul computes out[i] = a[i] * b[i]
func ElemwiseMul(out, a, b []float32, size int) {
	for i := 0; i < size; i++ {
		out[i] = a[i] * b[i]
	}
}

// VecAdd computes a[i] += b[i] in-place.
func VecAdd(a, b []float32, size int) {
	for i := 0; i < size; i++ {
		a[i] += b[i]
	}
}
