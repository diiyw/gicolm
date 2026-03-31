package main

import (
	"math"
	"sort"
)

type Sampler struct {
	Temperature float32
	TopP        float32
	RngState    uint64
}

type probIndex struct {
	Prob  float32
	Index int
}

func xorshift64(state *uint64) uint64 {
	x := *state
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*state = x
	return x
}

func randFloat(state *uint64) float32 {
	return float32(xorshift64(state)>>11) / float32(1<<53)
}

func SamplerInit(s *Sampler, temperature, topP float32, seed uint64) {
	s.Temperature = temperature
	s.TopP = topP
	if seed != 0 {
		s.RngState = seed
	} else {
		s.RngState = 42
	}
}

func SamplerSample(s *Sampler, logits []float32, vocabSize int) int {
	// Greedy (temperature 0)
	if s.Temperature <= 0.0 {
		best := 0
		for i := 1; i < vocabSize; i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}

	// Apply temperature
	invTemp := 1.0 / s.Temperature
	for i := 0; i < vocabSize; i++ {
		logits[i] *= invTemp
	}

	// Softmax
	Softmax(logits, vocabSize)

	// If top_p >= 1.0, sample from full distribution
	if s.TopP >= 1.0 {
		r := randFloat(&s.RngState)
		var cum float32
		for i := 0; i < vocabSize; i++ {
			cum += logits[i]
			if cum > r {
				return i
			}
		}
		return vocabSize - 1
	}

	// Top-p (nucleus) sampling
	sorted := make([]probIndex, vocabSize)
	for i := 0; i < vocabSize; i++ {
		sorted[i] = probIndex{Prob: logits[i], Index: i}
	}
	sort.Slice(sorted, func(a, b int) bool {
		return sorted[a].Prob > sorted[b].Prob
	})

	// Find cutoff where cumulative probability exceeds top_p
	var cum float32
	cutoff := 0
	for i := 0; i < vocabSize; i++ {
		cum += sorted[i].Prob
		cutoff = i + 1
		if cum >= s.TopP {
			break
		}
	}

	// Sample from truncated distribution
	r := randFloat(&s.RngState) * cum
	var acc float32
	result := sorted[0].Index
	for i := 0; i < cutoff; i++ {
		acc += sorted[i].Prob
		if acc > r {
			result = sorted[i].Index
			break
		}
	}

	return result
}

func exp32(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
