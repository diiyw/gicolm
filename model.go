package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
)

const (
	GGUFMagic    uint32 = 0x46554747
	MaxLayers           = 64
	KVCacheMagic uint32 = 0x4B564350
)

type ModelConfig struct {
	NEmbd        int
	NFfn         int
	NHeads       int
	NKVHeads     int
	NLayers      int
	VocabSize    int
	MaxSeqLen    int
	HeadDim      int
	RopeFreqBase float32
	Alignment    int
	WeightType   GGUFType
}

type LayerWeights struct {
	AttnNorm       []byte
	AttnQ          []byte
	AttnK          []byte
	AttnV          []byte
	AttnOutput     []byte
	FfnNorm        []byte
	FfnGate        []byte
	FfnDown        []byte
	FfnUp          []byte
	TypeAttnNorm   GGUFType
	TypeAttnQ      GGUFType
	TypeAttnK      GGUFType
	TypeAttnV      GGUFType
	TypeAttnOutput GGUFType
	TypeFfnNorm    GGUFType
	TypeFfnGate    GGUFType
	TypeFfnDown    GGUFType
	TypeFfnUp      GGUFType
}

type ModelWeights struct {
	TokenEmbd      []byte
	TypeTokenEmbd  GGUFType
	OutputNorm     []byte
	TypeOutputNorm GGUFType
	Output         []byte
	TypeOutput     GGUFType
	Layers         [MaxLayers]LayerWeights
}

type RunState struct {
	X              []float32
	Xb             []float32
	Xb2            []float32
	Q              []float32
	Hb             []float32
	Hb2            []float32
	Logits         []float32
	DequantScratch []float32

	KeyCache []uint16
	ValCache []uint16

	RopeCos []float32
	RopeSin []float32

	NormWeights []float32
	AttnNormW   [MaxLayers][]float32
	FfnNormW    [MaxLayers][]float32
	OutputNormW []float32
}

type Model struct {
	Config  ModelConfig
	Weights ModelWeights
	State   RunState

	Data []byte // mmap'd file data

	TokTokensData []byte
	TokNTokens    uint64
	TokScoresData []byte
	TokNScores    uint64
	TokBosID      uint32
	TokEosID      uint32
}

// GGUF metadata value types
const (
	ggufMetaUint8   = 0
	ggufMetaInt8    = 1
	ggufMetaUint16  = 2
	ggufMetaInt16   = 3
	ggufMetaUint32  = 4
	ggufMetaInt32   = 5
	ggufMetaFloat32 = 6
	ggufMetaBool    = 7
	ggufMetaString  = 8
	ggufMetaArray   = 9
	ggufMetaUint64  = 10
	ggufMetaInt64   = 11
	ggufMetaFloat64 = 12
)

type reader struct {
	data []byte
	pos  int
}

func (r *reader) readU8() byte {
	v := r.data[r.pos]
	r.pos++
	return v
}

func (r *reader) readU16() uint16 {
	v := binary.LittleEndian.Uint16(r.data[r.pos:])
	r.pos += 2
	return v
}

func (r *reader) readU32() uint32 {
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return v
}

func (r *reader) readI32() int32 {
	return int32(r.readU32())
}

func (r *reader) readU64() uint64 {
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return v
}

func (r *reader) readF32() float32 {
	return math.Float32frombits(r.readU32())
}

func (r *reader) readGGUFString() string {
	slen := r.readU64()
	s := string(r.data[r.pos : r.pos+int(slen)])
	r.pos += int(slen)
	return s
}

func (r *reader) skipMetaValue(vtype uint32) {
	switch vtype {
	case ggufMetaUint8, ggufMetaInt8, ggufMetaBool:
		r.pos++
	case ggufMetaUint16, ggufMetaInt16:
		r.pos += 2
	case ggufMetaUint32, ggufMetaInt32, ggufMetaFloat32:
		r.pos += 4
	case ggufMetaUint64, ggufMetaInt64:
		r.pos += 8
	case ggufMetaFloat64:
		r.pos += 8
	case ggufMetaString:
		r.readGGUFString()
	case ggufMetaArray:
		arrType := r.readU32()
		arrLen := r.readU64()
		for i := uint64(0); i < arrLen; i++ {
			r.skipMetaValue(arrType)
		}
	}
}

// readMetaInt reads a numeric metadata value and advances position.
// Returns the value and whether it was numeric.
func (r *reader) readMetaInt(vtype uint32) (uint64, bool) {
	switch vtype {
	case ggufMetaUint8:
		v := r.data[r.pos]
		r.pos++
		return uint64(v), true
	case ggufMetaInt8:
		v := int8(r.data[r.pos])
		r.pos++
		return uint64(int64(v)), true
	case ggufMetaUint16:
		v := binary.LittleEndian.Uint16(r.data[r.pos:])
		r.pos += 2
		return uint64(v), true
	case ggufMetaInt16:
		v := int16(binary.LittleEndian.Uint16(r.data[r.pos:]))
		r.pos += 2
		return uint64(int64(v)), true
	case ggufMetaUint32:
		v := binary.LittleEndian.Uint32(r.data[r.pos:])
		r.pos += 4
		return uint64(v), true
	case ggufMetaInt32:
		v := int32(binary.LittleEndian.Uint32(r.data[r.pos:]))
		r.pos += 4
		return uint64(int64(v)), true
	case ggufMetaUint64:
		v := binary.LittleEndian.Uint64(r.data[r.pos:])
		r.pos += 8
		return v, true
	case ggufMetaInt64:
		v := int64(binary.LittleEndian.Uint64(r.data[r.pos:]))
		r.pos += 8
		return uint64(v), true
	case ggufMetaFloat32:
		r.pos += 4
		return 0, false
	case ggufMetaFloat64:
		r.pos += 8
		return 0, false
	case ggufMetaBool:
		v := r.data[r.pos]
		r.pos++
		return uint64(v), true
	case ggufMetaString:
		r.readGGUFString()
		return 0, false
	case ggufMetaArray:
		arrType := r.readU32()
		arrLen := r.readU64()
		for i := uint64(0); i < arrLen; i++ {
			r.skipMetaValue(arrType)
		}
		return 0, false
	default:
		return 0, false
	}
}

func mmapFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := fi.Size()
	if size == 0 {
		return nil, fmt.Errorf("empty file")
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap failed: %w", err)
	}

	return data, nil
}

func munmapFile(data []byte) error {
	if data != nil {
		return syscall.Munmap(data)
	}
	return nil
}

func parseGGUF(m *Model, maxSeqLen int) error {
	r := &reader{data: m.Data, pos: 0}
	cfg := &m.Config

	magic := r.readU32()
	if magic != GGUFMagic {
		return fmt.Errorf("invalid GGUF magic: 0x%08X", magic)
	}

	version := r.readU32()
	if version < 2 || version > 3 {
		return fmt.Errorf("unsupported GGUF version: %d", version)
	}

	nTensors := r.readU64()
	nMetadata := r.readU64()

	cfg.Alignment = 32
	cfg.RopeFreqBase = 10000.0
	cfg.MaxSeqLen = 2048
	cfg.WeightType = GGUFTypeF16
	m.TokBosID = 1
	m.TokEosID = 2

	for i := uint64(0); i < nMetadata; i++ {
		key := r.readGGUFString()
		vtype := r.readU32()

		switch key {
		case "llama.embedding_length", "general.embedding_length":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.NEmbd = int(v)
			}
		case "llama.feed_forward_length", "general.feed_forward_length":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.NFfn = int(v)
			}
		case "llama.attention.head_count":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.NHeads = int(v)
			}
		case "llama.attention.head_count_kv":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.NKVHeads = int(v)
			}
		case "llama.block_count":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.NLayers = int(v)
			}
		case "llama.context_length":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.MaxSeqLen = int(v)
			}
		case "llama.rope.freq_base":
			if vtype == ggufMetaFloat32 {
				cfg.RopeFreqBase = r.readF32()
			} else {
				r.skipMetaValue(vtype)
			}
		case "general.alignment":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.Alignment = int(v)
			}
		case "llama.vocab_size":
			if v, ok := r.readMetaInt(vtype); ok {
				cfg.VocabSize = int(v)
			}
		case "tokenizer.ggml.bos_token_id":
			if v, ok := r.readMetaInt(vtype); ok {
				m.TokBosID = uint32(v)
			}
		case "tokenizer.ggml.eos_token_id":
			if v, ok := r.readMetaInt(vtype); ok {
				m.TokEosID = uint32(v)
			}
		case "tokenizer.ggml.tokens":
			if vtype != ggufMetaArray {
				r.skipMetaValue(vtype)
			} else {
				arrType := r.readU32()
				arrLen := r.readU64()
				m.TokTokensData = r.data[r.pos:]
				m.TokNTokens = arrLen
				for j := uint64(0); j < arrLen; j++ {
					r.skipMetaValue(arrType)
				}
			}
		case "tokenizer.ggml.scores":
			if vtype != ggufMetaArray {
				r.skipMetaValue(vtype)
			} else {
				arrType := r.readU32()
				arrLen := r.readU64()
				_ = arrType
				m.TokScoresData = r.data[r.pos:]
				m.TokNScores = arrLen
				r.pos += int(arrLen) * 4
			}
		default:
			r.skipMetaValue(vtype)
		}
	}

	if maxSeqLen > 0 && maxSeqLen < cfg.MaxSeqLen {
		cfg.MaxSeqLen = maxSeqLen
	}
	cfg.HeadDim = cfg.NEmbd / cfg.NHeads

	// Parse tensor info entries
	type tensorInfo struct {
		Name   string
		NDims  uint32
		Dims   [4]uint64
		Type   uint32
		Offset uint64
	}

	tinfos := make([]tensorInfo, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		tinfos[i].Name = r.readGGUFString()
		tinfos[i].NDims = r.readU32()
		for d := uint32(0); d < tinfos[i].NDims; d++ {
			tinfos[i].Dims[d] = r.readU64()
		}
		tinfos[i].Type = r.readU32()
		tinfos[i].Offset = r.readU64()
	}

	alignment := cfg.Alignment
	tensorDataBase := (r.pos + alignment - 1) & ^(alignment - 1)

	w := &m.Weights

	for i := uint64(0); i < nTensors; i++ {
		ptr := m.Data[tensorDataBase+int(tinfos[i].Offset):]
		qtype := GGUFType(tinfos[i].Type)

		switch {
		case tinfos[i].Name == "token_embd.weight":
			w.TokenEmbd = ptr
			w.TypeTokenEmbd = qtype
		case tinfos[i].Name == "output_norm.weight":
			w.OutputNorm = ptr
			w.TypeOutputNorm = qtype
		case tinfos[i].Name == "output.weight":
			w.Output = ptr
			w.TypeOutput = qtype
		default:
			layer := -1
			suffix := ""

			if len(tinfos[i].Name) > 4 && tinfos[i].Name[:4] == "blk." {
				p := 4
				end := len(tinfos[i].Name)
				layer = 0
				for p < end && tinfos[i].Name[p] >= '0' && tinfos[i].Name[p] <= '9' {
					layer = layer*10 + int(tinfos[i].Name[p]-'0')
					p++
				}
				if p < end && tinfos[i].Name[p] == '.' {
					p++
					suffix = tinfos[i].Name[p:]
				}
			}

			if layer >= 0 && layer < MaxLayers {
				lw := &w.Layers[layer]
				switch suffix {
				case "attn_norm.weight":
					lw.AttnNorm = ptr
					lw.TypeAttnNorm = qtype
				case "attn_q.weight":
					lw.AttnQ = ptr
					lw.TypeAttnQ = qtype
				case "attn_k.weight":
					lw.AttnK = ptr
					lw.TypeAttnK = qtype
				case "attn_v.weight":
					lw.AttnV = ptr
					lw.TypeAttnV = qtype
				case "attn_output.weight":
					lw.AttnOutput = ptr
					lw.TypeAttnOutput = qtype
				case "ffn_norm.weight":
					lw.FfnNorm = ptr
					lw.TypeFfnNorm = qtype
				case "ffn_gate.weight":
					lw.FfnGate = ptr
					lw.TypeFfnGate = qtype
				case "ffn_down.weight":
					lw.FfnDown = ptr
					lw.TypeFfnDown = qtype
				case "ffn_up.weight":
					lw.FfnUp = ptr
					lw.TypeFfnUp = qtype
				}
			}
		}
	}

	if len(w.Output) == 0 {
		w.Output = w.TokenEmbd
		w.TypeOutput = w.TypeTokenEmbd
	}

	if cfg.VocabSize == 0 {
		for i := uint64(0); i < nTensors; i++ {
			if tinfos[i].Name == "token_embd.weight" {
				if tinfos[i].NDims >= 2 {
					d0 := int(tinfos[i].Dims[0])
					d1 := int(tinfos[i].Dims[1])
					if d0 == cfg.NEmbd {
						cfg.VocabSize = d1
					} else {
						cfg.VocabSize = d0
					}
				}
				break
			}
		}
	}
	if cfg.VocabSize == 0 && m.TokNTokens > 0 {
		cfg.VocabSize = int(m.TokNTokens)
	}

	cfg.WeightType = w.Layers[0].TypeAttnQ

	fmt.Fprintf(os.Stderr, "Model config:\n")
	fmt.Fprintf(os.Stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
		cfg.NEmbd, cfg.NFfn, cfg.NHeads, cfg.NKVHeads)
	fmt.Fprintf(os.Stderr, "  n_layers=%d, vocab_size=%d, max_seq=%d\n",
		cfg.NLayers, cfg.VocabSize, cfg.MaxSeqLen)
	fmt.Fprintf(os.Stderr, "  head_dim=%d, rope_base=%.1f\n", cfg.HeadDim, cfg.RopeFreqBase)

	return nil
}

func initRopeTables(s *RunState, c *ModelConfig) {
	halfDim := c.HeadDim / 2
	for pos := 0; pos < c.MaxSeqLen; pos++ {
		cosRow := s.RopeCos[pos*halfDim:]
		sinRow := s.RopeSin[pos*halfDim:]
		for i := 0; i < halfDim; i++ {
			theta := float32(pos) / float32(math.Pow(float64(c.RopeFreqBase), float64(2*i)/float64(c.HeadDim)))
			cosRow[i] = float32(math.Cos(float64(theta)))
			sinRow[i] = float32(math.Sin(float64(theta)))
		}
	}
}

func allocateRunState(m *Model) error {
	c := &m.Config
	s := &m.State

	kvDim := c.NKVHeads * c.HeadDim
	halfDim := c.HeadDim / 2

	s.X = make([]float32, c.NEmbd)
	s.Xb = make([]float32, c.NEmbd)
	s.Xb2 = make([]float32, c.NEmbd)
	s.Q = make([]float32, c.NEmbd)
	s.Hb = make([]float32, c.NFfn)
	s.Hb2 = make([]float32, c.NFfn)
	s.Logits = make([]float32, c.VocabSize)

	scratchDim := c.NEmbd
	if c.NFfn > scratchDim {
		scratchDim = c.NFfn
	}
	if c.VocabSize > scratchDim {
		scratchDim = c.VocabSize
	}
	s.DequantScratch = make([]float32, scratchDim)

	s.RopeCos = make([]float32, c.MaxSeqLen*halfDim)
	s.RopeSin = make([]float32, c.MaxSeqLen*halfDim)

	normCount := (c.NLayers*2 + 1) * c.NEmbd
	s.NormWeights = make([]float32, normCount)

	// FP16 KV cache
	kvElements := c.NLayers * c.MaxSeqLen * kvDim
	s.KeyCache = make([]uint16, kvElements)
	s.ValCache = make([]uint16, kvElements)

	// Pre-dequantize norm weights
	nw := s.NormWeights
	for l := 0; l < c.NLayers; l++ {
		s.AttnNormW[l] = nw[l*c.NEmbd : (l+1)*c.NEmbd]
		DequantizeRow(m.Weights.Layers[l].AttnNorm, s.AttnNormW[l], c.NEmbd, m.Weights.Layers[l].TypeAttnNorm)

		off := (c.NLayers + l) * c.NEmbd
		s.FfnNormW[l] = nw[off : off+c.NEmbd]
		DequantizeRow(m.Weights.Layers[l].FfnNorm, s.FfnNormW[l], c.NEmbd, m.Weights.Layers[l].TypeFfnNorm)
	}
	off := 2 * c.NLayers * c.NEmbd
	s.OutputNormW = nw[off : off+c.NEmbd]
	DequantizeRow(m.Weights.OutputNorm, s.OutputNormW, c.NEmbd, m.Weights.TypeOutputNorm)

	initRopeTables(s, c)

	return nil
}

func ModelLoad(m *Model, path string, maxSeqLen int) error {
	data, err := mmapFile(path)
	if err != nil {
		return fmt.Errorf("mmap failed: %w", err)
	}
	m.Data = data

	if err := parseGGUF(m, maxSeqLen); err != nil {
		munmapFile(data)
		return err
	}

	if err := allocateRunState(m); err != nil {
		munmapFile(data)
		return err
	}

	return nil
}

// ModelForward runs one forward pass and returns the logits.
func ModelForward(m *Model, token, pos int) []float32 {
	c := &m.Config
	w := &m.Weights
	s := &m.State

	dim := c.NEmbd
	nFfn := c.NFfn
	nHeads := c.NHeads
	nKVHeads := c.NKVHeads
	headDim := c.HeadDim
	kvDim := nKVHeads * headDim
	kvMul := nHeads / nKVHeads
	seqLen := c.MaxSeqLen
	halfDim := headDim / 2

	// RoPE tables for this position
	cosPos := s.RopeCos[pos*halfDim:]
	sinPos := s.RopeSin[pos*halfDim:]

	// 1. Embedding lookup
	{
		rowBytes := GGUFTypeRowSize(w.TypeTokenEmbd, dim)
		embdRow := w.TokenEmbd[token*rowBytes:]
		DequantizeRow(embdRow, s.X, dim, w.TypeTokenEmbd)
	}

	// 2. Transformer layers
	for l := 0; l < c.NLayers; l++ {
		lw := &w.Layers[l]

		// Attention
		RmsNorm(s.Xb, s.X, s.AttnNormW[l], dim)

		// QKV projections
		MatMul(s.Q, s.Xb, lw.AttnQ, dim, dim, lw.TypeAttnQ)

		// K projection
		kTmp := s.Xb2
		MatMul(kTmp, s.Xb, lw.AttnK, dim, kvDim, lw.TypeAttnK)

		// K cache offset
		kcacheLayer := s.KeyCache[l*seqLen*kvDim:]
		vcacheLayer := s.ValCache[l*seqLen*kvDim:]
		keyPosFp16 := kcacheLayer[pos*kvDim:]

		// Apply RoPE to Q and K
		RoPE(s.Q, kTmp, headDim, nHeads, nKVHeads, cosPos, sinPos)

		// Convert K to FP16 and store
		for d := 0; d < kvDim; d++ {
			keyPosFp16[d] = fp32ToFp16(kTmp[d])
		}

		// V projection -> store as FP16
		vTmp := s.Xb2
		MatMul(vTmp, s.Xb, lw.AttnV, dim, kvDim, lw.TypeAttnV)
		valPosFp16 := vcacheLayer[pos*kvDim:]
		for d := 0; d < kvDim; d++ {
			valPosFp16[d] = fp32ToFp16(vTmp[d])
		}

		// Flash attention (online softmax)
		for h := 0; h < nHeads; h++ {
			qh := s.Q[h*headDim:]
			kvH := h / kvMul
			xbh := s.Xb[h*headDim:]

			maxScore := float32(-1e30)
			var sumExp float32
			acc := make([]float32, headDim)

			for t := 0; t <= pos; t++ {
				// Compute score: dot(Q_h, K_t) / sqrt(head_dim)
				ktBase := t*kvDim + kvH*headDim
				var score float32
				for d := 0; d < headDim; d++ {
					score += qh[d] * fp16ToFp32(kcacheLayer[ktBase+d])
				}
				score /= float32(math.Sqrt(float64(headDim)))

				// Online softmax update
				vtBase := t*kvDim + kvH*headDim
				if score > maxScore {
					correction := float32(math.Exp(float64(maxScore - score)))
					sumExp = sumExp*correction + 1.0
					for d := 0; d < headDim; d++ {
						acc[d] = acc[d]*correction + fp16ToFp32(vcacheLayer[vtBase+d])
					}
					maxScore = score
				} else {
					wt := float32(math.Exp(float64(score - maxScore)))
					sumExp += wt
					for d := 0; d < headDim; d++ {
						acc[d] += wt * fp16ToFp32(vcacheLayer[vtBase+d])
					}
				}
			}

			// Normalize
			invSum := 1.0 / sumExp
			for d := 0; d < headDim; d++ {
				xbh[d] = acc[d] * invSum
			}
		}

		// Output projection
		MatMul(s.Xb2, s.Xb, lw.AttnOutput, dim, dim, lw.TypeAttnOutput)
		VecAdd(s.X, s.Xb2, dim)

		// FFN (SwiGLU)
		RmsNorm(s.Xb, s.X, s.FfnNormW[l], dim)

		MatMul(s.Hb, s.Xb, lw.FfnGate, dim, nFfn, lw.TypeFfnGate)
		MatMul(s.Hb2, s.Xb, lw.FfnUp, dim, nFfn, lw.TypeFfnUp)

		SiLU(s.Hb, nFfn)
		ElemwiseMul(s.Hb, s.Hb, s.Hb2, nFfn)

		MatMul(s.Xb, s.Hb, lw.FfnDown, nFfn, dim, lw.TypeFfnDown)
		VecAdd(s.X, s.Xb, dim)
	}

	// 3. Final RMSNorm
	RmsNorm(s.X, s.X, s.OutputNormW, dim)

	// 4. Output projection -> logits
	MatMul(s.Logits, s.X, w.Output, dim, c.VocabSize, w.TypeOutput)

	return s.Logits
}

func ModelFree(m *Model) {
	if m.Data != nil {
		munmapFile(m.Data)
		m.Data = nil
	}
}

// KV Cache save/load

func KVCacheSave(m *Model, path string, nPos int) error {
	c := &m.Config
	kvDim := c.NKVHeads * c.HeadDim
	seqLen := c.MaxSeqLen

	if nPos <= 0 || nPos > seqLen {
		return fmt.Errorf("invalid n_pos")
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("kvcache_save: cannot open %s: %w", path, err)
	}
	defer f.Close()

	// Write header
	header := [4]uint32{KVCacheMagic, uint32(nPos), uint32(c.NLayers), uint32(kvDim)}
	binary.Write(f, binary.LittleEndian, header)

	// Write KV cache
	for l := 0; l < c.NLayers; l++ {
		kcacheL := m.State.KeyCache[l*seqLen*kvDim:]
		for p := 0; p < nPos; p++ {
			binary.Write(f, binary.LittleEndian, kcacheL[p*kvDim:(p+1)*kvDim])
		}
	}
	for l := 0; l < c.NLayers; l++ {
		vcacheL := m.State.ValCache[l*seqLen*kvDim:]
		for p := 0; p < nPos; p++ {
			binary.Write(f, binary.LittleEndian, vcacheL[p*kvDim:(p+1)*kvDim])
		}
	}

	fmt.Fprintf(os.Stderr, "KV cache saved: %d positions to %s\n", nPos, path)
	return nil
}

func KVCacheLoad(m *Model, path string) int {
	c := &m.Config
	kvDim := c.NKVHeads * c.HeadDim
	seqLen := c.MaxSeqLen

	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()

	var header [4]uint32
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		return 0
	}

	if header[0] != KVCacheMagic {
		fmt.Fprintf(os.Stderr, "kvcache_load: invalid magic\n")
		return 0
	}

	nPos := int(header[1])
	fileLayers := int(header[2])
	fileKvDim := int(header[3])

	if fileLayers != c.NLayers || fileKvDim != kvDim {
		fmt.Fprintf(os.Stderr, "kvcache_load: model mismatch\n")
		return 0
	}
	if nPos > seqLen {
		fmt.Fprintf(os.Stderr, "kvcache_load: cached positions exceeds max_seq_len\n")
		return 0
	}

	for l := 0; l < c.NLayers; l++ {
		kcacheL := m.State.KeyCache[l*seqLen*kvDim:]
		for p := 0; p < nPos; p++ {
			if err := binary.Read(f, binary.LittleEndian, kcacheL[p*kvDim:(p+1)*kvDim]); err != nil {
				return 0
			}
		}
	}
	for l := 0; l < c.NLayers; l++ {
		vcacheL := m.State.ValCache[l*seqLen*kvDim:]
		for p := 0; p < nPos; p++ {
			if err := binary.Read(f, binary.LittleEndian, vcacheL[p*kvDim:(p+1)*kvDim]); err != nil {
				return 0
			}
		}
	}

	fmt.Fprintf(os.Stderr, "KV cache loaded: %d positions from %s\n", nPos, path)
	return nPos
}
