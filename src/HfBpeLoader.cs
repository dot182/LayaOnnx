using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace LayaOnnx;

/// <summary>
/// Laya's checkpoints ship a single HuggingFace-style "tokenizer.json" (the "fast tokenizer" format),
/// not the separate vocab.json / merges.txt files that Microsoft.ML.Tokenizers' BpeTokenizer.Create(string, string)
/// expects. This loader pulls the "model.vocab" / "model.merges" / "added_tokens" sections out of that single
/// file and feeds them into Microsoft.ML.Tokenizers directly.
///
/// This targets the English Laya checkpoint (ModernBERT), whose tokenizer is byte-level BPE, the same family as
/// GPT-2/RoBERTa - so it uses Microsoft.ML.Tokenizers' built-in RobertaPreTokenizer. If you point this at the
/// multilingual checkpoint (mmBERT, a SentencePiece-style tokenizer) you'll need to pass a different PreTokenizer
/// (or use Microsoft.ML.Tokenizers' SentencePiece support) via the optional parameter below.
/// </summary>
public static class HfBpeLoader
{
    public sealed class LoadedTokenizer
    {
        public required BpeTokenizer Tokenizer { get; init; }

        /// <summary>Every token string the tokenizer knows about (base vocab + added/special tokens), used to
        /// resolve special-token names (e.g. "[MASK]", "&lt;mask&gt;") to ids.</summary>
        public required IReadOnlyDictionary<string, int> TokenToId { get; init; }
    }

    public static LoadedTokenizer Load(string tokenizerJsonPath, PreTokenizer? preTokenizer = null)
    {
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var model = root.GetProperty("model");
        var modelType = model.TryGetProperty("type", out var typeElem) ? typeElem.GetString() : null;
        if (modelType is not null && modelType != "BPE")
        {
            throw new NotSupportedException(
                $"'{tokenizerJsonPath}' uses tokenizer model type '{modelType}', but this loader only handles " +
                "'BPE'. The Laya multilingual (mmBERT) checkpoint uses a different tokenizer - see the class " +
                "remarks for how to adapt this.");
        }

        var vocab = new Dictionary<string, int>();
        foreach (var entry in model.GetProperty("vocab").EnumerateObject())
        {
            vocab[entry.Name] = entry.Value.GetInt32();
        }

        var merges = new List<string>();
        if (model.TryGetProperty("merges", out var mergesElem))
        {
            foreach (var m in mergesElem.EnumerateArray())
            {
                // Newer tokenizers.json versions store each merge as a 2-element array instead of "a b".
                merges.Add(m.ValueKind == JsonValueKind.Array
                    ? string.Join(' ', m.EnumerateArray().Select(x => x.GetString()))
                    : m.GetString()!);
            }
        }

        string? unknownToken = model.TryGetProperty("unk_token", out var unkElem) && unkElem.ValueKind == JsonValueKind.String
            ? unkElem.GetString()
            : null;

        // "added_tokens" holds things like [CLS]/[SEP]/[MASK]/[PAD] (or <bos>/<eos>/<mask>/<pad> for other
        // checkpoints) - they're not always duplicated inside model.vocab, so collect them separately too.
        var specialTokens = new Dictionary<string, int>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var tok in addedTokens.EnumerateArray())
            {
                var content = tok.GetProperty("content").GetString()!;
                var id = tok.GetProperty("id").GetInt32();
                specialTokens[content] = id;
                vocab.TryAdd(content, id);
            }
        }

        var options = new BpeOptions(vocab)
        {
            Merges = merges,
            SpecialTokens = specialTokens.Count > 0 ? specialTokens : null,
            UnknownToken = unknownToken,
            PreTokenizer = preTokenizer ?? RobertaPreTokenizer.Instance,
        };

        var tokenizer = BpeTokenizer.Create(options);

        return new LoadedTokenizer { Tokenizer = tokenizer, TokenToId = vocab };
    }
}
