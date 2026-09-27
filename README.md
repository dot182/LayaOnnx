# LayaOnnx

A minimal C# / ONNX Runtime wrapper around [Laya](https://github.com/NandhaKishorM/laya), using
`Microsoft.ML.Tokenizers` for tokenization. You supply the paths to an exported `.onnx` checkpoint and its
`tokenizer.json`; model downloading, exporting, quantizing etc. are entirely up to you.

## Example Usage

```csharp
using var laya = new LayaModel("laya.onnx", "tokenizer.json");

var intent = laya.AskChoice(
    state: "Hi, my order never arrived and it's been two weeks. I'd like my money back.",
    instructions: "What does the customer want?",
    options: new[] { "refund", "replacement", "order status update", "general complaint" });

Console.WriteLine($"{intent.BestOption} ({intent.Confidence:P1})");
// or: laya.AskYesNo(state, "Is the customer frustrated?")
// or: laya.AskScore(state, "How severe is this ticket?", new[] { "low", "medium", "high", "critical" })
```

## What you need to provide

1. **The ONNX model** — export it yourself from a `convaiinnovations/laya*` checkpoint (the repo's
   `export_onnx.py`/similar scripts), or use one of the community ONNX conversions floating around Hugging
   Face. The graph must expose the standard Laya I/O contract:
   - Inputs: `input_ids`, `attention_mask` (`int64 [1, L]`), `marker_pos` (`int64 [1, k]`),
     `marker_mask` (`bool [1, k]`), `qtype` (`int64 [1]`, 0=choice/1=score/2=noul)
   - Outputs: `logits` (`float32 [1, k]`) — this wrapper only reads that one output.
2. **That checkpoint's `tokenizer.json`.**

## Tokenizer notes

`Microsoft.ML.Tokenizers`' `BpeTokenizer` normally loads a separate `vocab.json` + `merges.txt`. Laya ships a
single unified `tokenizer.json` instead, so `HfBpeLoader.cs` pulls the `model.vocab` / `model.merges` /
`added_tokens` sections out of that file itself and feeds them into `BpeOptions`.

This defaults to `RobertaPreTokenizer` (byte-level BPE), which matches the **English** Laya checkpoint
(ModernBERT-based, GPT-2/RoBERTa-style tokenizer). The **multilingual** checkpoint (mmBERT) uses a
SentencePiece-style tokenizer instead — if you're using that one, pass a matching `PreTokenizer` via
`LayaModel`'s constructor (or extend `HfBpeLoader`), or use `Microsoft.ML.Tokenizers`' `SentencePieceTokenizer`
support directly.

Special tokens (start/separator/mask) are auto-detected by name — it tries `[CLS]`/`[SEP]`/`[MASK]` (ModernBERT)
then `<bos>`/`<eos>`/`<mask>` (mmBERT/Gemma-style). Override them explicitly in the constructor if your
checkpoint uses different names.

## Calibration temperature

Laya's own SDK divides logits by a fitted "temperature" (per question type and option count) before the
softmax, taken from the checkpoint's `rl_agent_config.json`. This wrapper defaults to a temperature of `1.0`
(uncalibrated but still argmax-correct — the same option wins either way). If you have the real values, pass
them in:

```csharp
using var laya = new LayaModel("laya.onnx", "tokenizer.json",
    temperature: (questionType, optionCount) => questionType switch
    {
        LayaQuestionType.Choice => 1.0,
        LayaQuestionType.Score  => 1.2,
        LayaQuestionType.YesNo  => 0.9,
        _ => 1.0,
    });
```

## Files

- `src/HfBpeLoader.cs` — parses a HF `tokenizer.json` into a `BpeTokenizer`.
- `src/LayaModel.cs` — builds Laya's prompt/marker sequence, runs the ONNX graph, returns `LayaAnswer`.
- `src/Program.cs` — tiny CLI demo: `dotnet run -- path/to/model.onnx path/to/tokenizer.json`.

## Caveats

- Batch size is fixed at 1 (one question per call) to keep this simple — batch multiple questions yourself if
  you need the throughput.
- `head_max_len` (the combined budget for the instructions + options portion, 192 tokens on the English/
  multilingual checkpoints per Laya's docs) isn't separately enforced here; only the overall `maxLen` (default
  512) and 48-tokens-per-option truncation are, matching Laya's own reference ONNX example. Raise an option
  count that overflows this at your own risk — pass a smaller `maxLen`/`optionMaxTokens` if you hit issues.
- Package versions in the `.csproj` are placeholders — bump them to whatever's current when you restore.