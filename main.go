package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

func usage() {
	fmt.Fprintf(os.Stderr, "PicoLLM — ultra-lightweight LLM inference engine\n\n")
	fmt.Fprintf(os.Stderr, "Usage: picolm <model.gguf> [options]\n")
	fmt.Fprintf(os.Stderr, "\nGeneration options:\n")
	fmt.Fprintf(os.Stderr, "  -p <prompt>    Input prompt (or pipe via stdin)\n")
	fmt.Fprintf(os.Stderr, "  -n <int>       Max tokens to generate (default: 256)\n")
	fmt.Fprintf(os.Stderr, "  -t <float>     Temperature (default: 0.8, 0=greedy)\n")
	fmt.Fprintf(os.Stderr, "  -k <float>     Top-p / nucleus sampling (default: 0.9)\n")
	fmt.Fprintf(os.Stderr, "  -s <int>       RNG seed (default: 42)\n")
	fmt.Fprintf(os.Stderr, "  -c <int>       Context length override\n")
	fmt.Fprintf(os.Stderr, "  -j <int>       Number of threads (default: num CPUs)\n")
	fmt.Fprintf(os.Stderr, "\nAdvanced options:\n")
	fmt.Fprintf(os.Stderr, "  --json         Grammar-constrained JSON output mode\n")
	fmt.Fprintf(os.Stderr, "  --cache <file> KV cache file (saves/loads prompt state)\n")
}

func readStdin() string {
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return ""
	}
	return string(data)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	modelPath := os.Args[1]
	promptText := ""
	maxTokens := 256
	temperature := 0.8
	topP := 0.9
	seed := int64(42)
	contextOverride := 0
	numThreads := 0
	jsonMode := false
	cacheFile := ""

	// Parse arguments manually (flags after model path)
	for i := 2; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "-p":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "-p requires a value\n")
				os.Exit(1)
			}
			i++
			promptText = os.Args[i]
		case "-n":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "-n requires a value\n")
				os.Exit(1)
			}
			i++
			maxTokens, _ = strconv.Atoi(os.Args[i])
		case "-t":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "-t requires a value\n")
				os.Exit(1)
			}
			i++
			temperature, _ = strconv.ParseFloat(os.Args[i], 64)
		case "-k":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "-k requires a value\n")
				os.Exit(1)
			}
			i++
			topP, _ = strconv.ParseFloat(os.Args[i], 64)
		case "-s":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "-s requires a value\n")
				os.Exit(1)
			}
			i++
			seed, _ = strconv.ParseInt(os.Args[i], 10, 64)
		case "-c":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "-c requires a value\n")
				os.Exit(1)
			}
			i++
			contextOverride, _ = strconv.Atoi(os.Args[i])
		case "-j":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "-j requires a value\n")
				os.Exit(1)
			}
			i++
			numThreads, _ = strconv.Atoi(os.Args[i])
		case "--json":
			jsonMode = true
		case "--cache":
			if i+1 >= len(os.Args) {
				fmt.Fprintf(os.Stderr, "--cache requires a value\n")
				os.Exit(1)
			}
			i++
			cacheFile = os.Args[i]
		default:
			fmt.Fprintf(os.Stderr, "Unknown option: %s\n", os.Args[i])
			usage()
			os.Exit(1)
		}
	}

	if numThreads > 0 {
		SetThreads(numThreads)
	}

	// Read prompt from stdin if not provided via -p
	if promptText == "" {
		fi, _ := os.Stdin.Stat()
		if fi != nil && (fi.Mode()&os.ModeCharDevice) == 0 {
			promptText = readStdin()
		}
	}

	if promptText == "" {
		fmt.Fprintf(os.Stderr, "No prompt provided. Use -p or pipe via stdin.\n")
		usage()
		os.Exit(1)
	}

	// Load model
	fmt.Fprintf(os.Stderr, "Loading model: %s\n", modelPath)
	var model Model
	if err := ModelLoad(&model, modelPath, contextOverride); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}

	// Load tokenizer
	var tokenizer Tokenizer
	if err := tokenizerLoad(&tokenizer, &model); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load tokenizer: %v\n", err)
		ModelFree(&model)
		os.Exit(1)
	}

	// Init sampler
	var sampler Sampler
	SamplerInit(&sampler, float32(temperature), float32(topP), uint64(seed))

	// Init grammar constraint
	var grammar GrammarState
	grammarMode := GrammarNone
	if jsonMode {
		grammarMode = GrammarJSON
	}
	GrammarInit(&grammar, grammarMode, &tokenizer)
	if jsonMode {
		fmt.Fprintf(os.Stderr, "JSON grammar mode enabled\n")
	}

	// Try to load KV cache
	cachePos := 0
	if cacheFile != "" {
		cachePos = KVCacheLoad(&model, cacheFile)
	}

	// Encode prompt
	maxPromptTokens := len(promptText) + 3
	promptTokens := make([]int, maxPromptTokens)
	nPrompt := TokenizerEncode(&tokenizer, promptText, promptTokens, maxPromptTokens, 1)

	// If cache covers part of the prompt, skip those positions
	startPos := 0
	if cachePos > 0 && cachePos <= nPrompt {
		startPos = cachePos
		fmt.Fprintf(os.Stderr, "Skipping %d cached prompt tokens\n", startPos)
	}

	fmt.Fprintf(os.Stderr, "Prompt: %d tokens, generating up to %d (temp=%.2f, top_p=%.2f, threads=%d)\n",
		nPrompt, maxTokens, temperature, topP, nThreads)
	fmt.Fprintf(os.Stderr, "---\n")

	// Generation loop
	totalGen := 0
	tStart := time.Now()
	var tFirstToken time.Time

	token := promptTokens[0]
	if startPos > 0 {
		token = promptTokens[startPos-1]
	}
	pos := startPos
	if pos > 0 {
		pos--
	}
	totalSteps := nPrompt + maxTokens
	if totalSteps > model.Config.MaxSeqLen {
		totalSteps = model.Config.MaxSeqLen
	}

	for ; pos < totalSteps; pos++ {
		if pos < startPos {
			token = promptTokens[pos]
			continue
		}

		// Forward pass
		logits := ModelForward(&model, token, pos)

		var next int
		if pos < nPrompt-1 {
			// Prefill: use next prompt token
			next = promptTokens[pos+1]
		} else {
			// Generation
			if pos == nPrompt-1 {
				tFirstToken = time.Now()
			}

			GrammarApply(&grammar, logits, model.Config.VocabSize)
			next = SamplerSample(&sampler, logits, model.Config.VocabSize)

			GrammarAdvance(&grammar, &tokenizer, next)

			piece := TokenizerDecode(&tokenizer, token, next)
			fmt.Print(piece)

			totalGen++

			if next == int(tokenizer.EosID) {
				break
			}
			if GrammarIsComplete(&grammar) {
				break
			}
		}

		token = next
	}

	fmt.Println()
	tEnd := time.Now()

	// Save KV cache if requested
	if cacheFile != "" && nPrompt > 0 {
		KVCacheSave(&model, cacheFile, nPrompt)
	}

	// Stats
	totalTime := tEnd.Sub(tStart).Seconds()
	if tFirstToken.IsZero() {
		tFirstToken = tEnd
	}
	genTime := tEnd.Sub(tFirstToken).Seconds()
	prefillTime := tFirstToken.Sub(tStart).Seconds()
	actualPrefill := nPrompt - startPos
	if actualPrefill < 0 {
		actualPrefill = 0
	}

	cached := ""
	if startPos > 0 {
		cached = " [partially cached]"
	}
	fmt.Fprintf(os.Stderr, "---\n")
	fmt.Fprintf(os.Stderr, "Prefill: %d tokens in %.2fs (%.1f tok/s)%s\n",
		actualPrefill, prefillTime,
		float64(actualPrefill)/prefillTime, cached)
	fmt.Fprintf(os.Stderr, "Generation: %d tokens in %.2fs (%.1f tok/s)\n",
		totalGen, genTime,
		float64(totalGen)/genTime)
	fmt.Fprintf(os.Stderr, "Total: %.2fs\n", totalTime)
}
