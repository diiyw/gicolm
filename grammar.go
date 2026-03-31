package main

const (
	NegInf       float32 = -1e30
	maxNestDepth         = 50
)

type GrammarMode int

const (
	GrammarNone GrammarMode = 0
	GrammarJSON GrammarMode = 1
)

type GrammarState struct {
	mode GrammarMode

	braceDepth   int
	bracketDepth int
	inString     bool
	escapeNext   bool
	started      bool

	tokenBraceDelta        []int8
	tokenBracketDelta      []int8
	tokenFirstByte         []byte
	tokenHasUnmatchedQuote []int8

	vocabSize int
	eosID     int
}

func analyzeToken(str string) (braceDelta, bracketDelta int8, hasUnmatchedQuote int8, firstByte byte) {
	if len(str) == 0 {
		return 0, 0, 0, 0
	}
	firstByte = str[0]
	var bd, bkd, quotes, escape int

	for _, c := range str {
		if escape != 0 {
			escape = 0
			continue
		}
		switch c {
		case '\\':
			escape = 1
		case '{':
			bd++
		case '}':
			bd--
		case '[':
			bkd++
		case ']':
			bkd--
		case '"':
			quotes++
		}
	}

	if bd > 127 {
		bd = 127
	}
	if bd < -128 {
		bd = -128
	}
	if bkd > 127 {
		bkd = 127
	}
	if bkd < -128 {
		bkd = -128
	}

	return int8(bd), int8(bkd), int8(quotes & 1), firstByte
}

func GrammarInit(g *GrammarState, mode GrammarMode, tok *Tokenizer) {
	g.mode = mode
	g.vocabSize = tok.VocabSize
	g.eosID = int(tok.EosID)

	if mode == GrammarNone {
		return
	}

	vs := tok.VocabSize
	g.tokenBraceDelta = make([]int8, vs)
	g.tokenBracketDelta = make([]int8, vs)
	g.tokenFirstByte = make([]byte, vs)
	g.tokenHasUnmatchedQuote = make([]int8, vs)

	for i := 0; i < vs; i++ {
		s := tok.Vocab[i]
		if s != "" {
			bd, bkd, uq, fb := analyzeToken(s)
			g.tokenBraceDelta[i] = bd
			g.tokenBracketDelta[i] = bkd
			g.tokenHasUnmatchedQuote[i] = uq
			g.tokenFirstByte[i] = fb
		}
	}
}

func GrammarApply(g *GrammarState, logits []float32, vocabSize int) {
	if g.mode == GrammarNone {
		return
	}

	totalDepth := g.braceDepth + g.bracketDepth

	if !g.started {
		for i := 0; i < vocabSize; i++ {
			fb := g.tokenFirstByte[i]
			if fb != '{' && fb != '[' && fb != ' ' && fb != '\n' {
				logits[i] = NegInf
			}
		}
		logits[g.eosID] = NegInf
		return
	}

	for i := 0; i < vocabSize; i++ {
		if i == g.eosID {
			continue
		}

		newBrace := g.braceDepth + int(g.tokenBraceDelta[i])
		newBracket := g.bracketDepth + int(g.tokenBracketDelta[i])
		newTotal := newBrace + newBracket

		if newBrace < 0 || newBracket < 0 {
			logits[i] = NegInf
			continue
		}

		if g.inString {
			continue
		}

		if newTotal > maxNestDepth {
			logits[i] = NegInf
			continue
		}
	}

	if totalDepth > 0 {
		logits[g.eosID] = NegInf
	}

	if totalDepth == 0 && g.started {
		maxLogit := logits[0]
		for i := 1; i < vocabSize; i++ {
			if logits[i] > maxLogit {
				maxLogit = logits[i]
			}
		}
		logits[g.eosID] = maxLogit + 5.0
	}
}

func GrammarAdvance(g *GrammarState, tok *Tokenizer, token int) {
	if g.mode == GrammarNone {
		return
	}
	if token < 0 || token >= g.vocabSize {
		return
	}

	str := tok.Vocab[token]
	for _, c := range str {
		if g.escapeNext {
			g.escapeNext = false
			continue
		}

		if g.inString {
			if c == '\\' {
				g.escapeNext = true
			} else if c == '"' {
				g.inString = false
			}
		} else {
			switch c {
			case '{':
				g.braceDepth++
				g.started = true
			case '}':
				g.braceDepth--
			case '[':
				g.bracketDepth++
				g.started = true
			case ']':
				g.bracketDepth--
			case '"':
				g.inString = true
			}
		}
	}
}

func GrammarIsComplete(g *GrammarState) bool {
	if g.mode == GrammarNone {
		return false
	}
	return g.started && g.braceDepth == 0 && g.bracketDepth == 0 && !g.inString
}
