package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

type Tokenizer struct {
	Vocab     []string
	Scores    []float32
	VocabSize int
	SortedIdx []int
	BosID     uint32
	EosID     uint32
}

func tokenizerLoad(t *Tokenizer, m *Model) error {
	vs := m.Config.VocabSize
	t.VocabSize = vs
	t.BosID = m.TokBosID
	t.EosID = m.TokEosID

	t.Vocab = make([]string, vs)
	t.Scores = make([]float32, vs)
	t.SortedIdx = make([]int, vs)

	// Read vocab strings from GGUF metadata array
	if len(m.TokTokensData) > 0 && m.TokNTokens > 0 {
		p := m.TokTokensData
		n := int(m.TokNTokens)
		if n > vs {
			n = vs
		}

		for i := 0; i < n; i++ {
			slen := binary.LittleEndian.Uint64(p[:8])
			p = p[8:]
			t.Vocab[i] = string(p[:slen])
			p = p[slen:]
		}
	}

	// Fill any remaining entries
	for i := 0; i < vs; i++ {
		if t.Vocab[i] == "" {
			t.Vocab[i] = ""
		}
	}

	// Read scores
	if len(m.TokScoresData) > 0 && m.TokNScores > 0 {
		n := int(m.TokNScores)
		if n > vs {
			n = vs
		}
		for i := 0; i < n; i++ {
			t.Scores[i] = math.Float32frombits(binary.LittleEndian.Uint32(m.TokScoresData[i*4:]))
		}
	}

	// Build sorted index
	for i := 0; i < vs; i++ {
		t.SortedIdx[i] = i
	}
	sort.Slice(t.SortedIdx, func(a, b int) bool {
		return t.Vocab[t.SortedIdx[a]] < t.Vocab[t.SortedIdx[b]]
	})

	fmt.Fprintf(os.Stderr, "Tokenizer loaded: %d tokens, bos=%d, eos=%d\n", vs, t.BosID, t.EosID)
	return nil
}

func vocabLookup(t *Tokenizer, str string) int {
	lo := 0
	hi := t.VocabSize - 1
	for lo <= hi {
		mid := (lo + hi) / 2
		idx := t.SortedIdx[mid]
		cmp := strings.Compare(t.Vocab[idx], str)
		if cmp == 0 {
			return idx
		} else if cmp < 0 {
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	return -1
}

func TokenizerEncode(t *Tokenizer, text string, tokens []int, maxTokens int, addBos int) int {
	nTokens := 0

	if addBos != 0 && nTokens < maxTokens {
		tokens[nTokens] = int(t.BosID)
		nTokens++
	}

	if text == "" {
		return nTokens
	}

	// SentencePiece: replace spaces with ▁ (U+2581)
	var norm strings.Builder
	// Add leading ▁
	norm.WriteRune('\u2581')

	for _, c := range text {
		if c == ' ' {
			norm.WriteRune('\u2581')
		} else {
			norm.WriteRune(c)
		}
	}
	normStr := norm.String()

	// Step 1: Convert to individual character tokens
	mergeBuf := make([]int, 0, len(normStr))

	for i := 0; i < len(normStr); {
		// Determine UTF-8 character length
		clen := 1
		c := normStr[i]
		if c >= 0xF0 {
			clen = 4
		} else if c >= 0xE0 {
			clen = 3
		} else if c >= 0xC0 {
			clen = 2
		}
		if i+clen > len(normStr) {
			clen = len(normStr) - i
		}

		// Look up this character in vocab
		charStr := normStr[i : i+clen]
		tok := vocabLookup(t, charStr)
		if tok >= 0 {
			mergeBuf = append(mergeBuf, tok)
			i += clen
		} else {
			// Fall back to byte token
			byteTok := fmt.Sprintf("<0x%02X>", byte(normStr[i]))
			tok = vocabLookup(t, byteTok)
			if tok >= 0 {
				mergeBuf = append(mergeBuf, tok)
			}
			i++
		}
	}

	// Step 2: BPE merge loop
	for len(mergeBuf) >= 2 {
		bestScore := float32(-1e30)
		bestIdx := -1
		bestTok := -1

		for i := 0; i < len(mergeBuf)-1; i++ {
			s1 := t.Vocab[mergeBuf[i]]
			s2 := t.Vocab[mergeBuf[i+1]]
			merged := s1 + s2

			tok := vocabLookup(t, merged)
			if tok >= 0 && t.Scores[tok] > bestScore {
				bestScore = t.Scores[tok]
				bestIdx = i
				bestTok = tok
			}
		}

		if bestIdx < 0 {
			break
		}

		mergeBuf[bestIdx] = bestTok
		mergeBuf = append(mergeBuf[:bestIdx+1], mergeBuf[bestIdx+2:]...)
	}

	// Copy to output
	for i := 0; i < len(mergeBuf) && nTokens < maxTokens; i++ {
		tokens[nTokens] = mergeBuf[i]
		nTokens++
	}

	return nTokens
}

func TokenizerDecode(t *Tokenizer, prevToken, token int) string {
	if token < 0 || token >= t.VocabSize {
		return ""
	}

	str := t.Vocab[token]

	// Handle byte tokens: <0xHH> -> single byte
	if len(str) == 6 && str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>' {
		var val byte
		for i := 3; i < 5; i++ {
			val <<= 4
			c := str[i]
			if c >= '0' && c <= '9' {
				val += c - '0'
			} else if c >= 'A' && c <= 'F' {
				val += c - 'A' + 10
			} else if c >= 'a' && c <= 'f' {
				val += c - 'a' + 10
			}
		}
		return string([]byte{val})
	}

	// Handle SentencePiece leading space marker ▁ -> " "
	if len(str) >= 3 && str[0] == '\xE2' && str[1] == '\x96' && str[2] == '\x81' {
		rest := str[3:]
		if prevToken == int(t.BosID) {
			// After BOS, strip the leading space
			return rest
		}
		return " " + rest
	}

	return str
}
