using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace LayaOnnx
{
    /// <summary>
    /// The three question types Laya's decision head supports. These map directly to the "qtype" graph
    /// input (0/1/2) - see https://github.com/NandhaKishorM/laya.
    /// </summary>
    public enum LayaQuestionType
    {
        /// <summary>Pick the best of several labelled options.</summary>
        Choice = 0,

        /// <summary>Score the state against an ordered list of options (e.g. a severity scale).</summary>
        Score = 1,

        /// <summary>A yes/no proposition - "is this statement true of the state?".</summary>
        YesNo = 2,
    }

    /// <summary>One option Laya can pick: a short <paramref name="Key"/> used as the answer identifier (e.g.
    /// "reship") plus the human-readable <paramref name="Description"/> shown to the model (e.g. "Send a
    /// replacement"). For plain string options the two are just the same value.</summary>
    public readonly struct LayaOption
    {
        public readonly string Key;
        public readonly string Description;

        public LayaOption(string key, string description)
        {
            Key = key;
            Description = description;
        }

        public LayaOption(string keyAndDescription)
        {
            Key = keyAndDescription;
            Description = keyAndDescription;
        }
    }
    /// <summary>One typed answer from Laya: a probability per option, plus the model's pick.</summary>
    public sealed class LayaAnswer
    {
        /// <summary>The option keys, in the same order as <see cref="Probabilities"/> (e.g. "reship", "notify").</summary>
        public string[] Options;

        /// <summary>The option descriptions shown to the model, same order as <see cref="Options"/>.</summary>
        public string[] Descriptions;

        /// <summary>Calibrated probability per option, same order as <see cref="Options"/>. Sums to 1.</summary>
        public double[] Probabilities;

        public int BestIndex;

        public string BestOption => Options[BestIndex];

        /// <summary>Probability of the picked option - Laya's own confidence in its answer.</summary>
        public double Confidence => Probabilities[BestIndex];

        /// <summary>For Score questions: the ordinal expectation Σ i·p(i) over the option indices.
        /// For YesNo questions, <see cref="Probabilities"/>[1] is P(true) directly. Ignore this for Choice.</summary>
        public double ExpectedScore;

        public override string ToString() => $"{BestOption} ({Confidence:P1})";
    }

    /// <summary>
    /// Minimal ONNX Runtime wrapper around a Laya checkpoint (https://github.com/NandhaKishorM/laya).
    ///
    /// You provide the paths; nothing is downloaded here. You need, next to each other or wherever you like:
    ///   - the exported ONNX model (e.g. laya.onnx / model.onnx, whatever export you used)
    ///   - that checkpoint's tokenizer.json
    ///
    /// This defaults to the English checkpoint's byte-level BPE tokenizer (ModernBERT / RoBERTa-style). If you're
    /// using the multilingual (mmBERT) checkpoint, its tokenizer is SentencePiece-based - pass a matching
    /// PreTokenizer via the constructor, or adapt HfBpeLoader.
    /// </summary>
    public sealed class LayaModel : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly BpeTokenizer _tokenizer;
        private readonly long _bosId;
        private readonly long _eosId;
        private readonly long _maskId;
        private readonly int _maxLen;
        private readonly int _optionMaxTokens;
        private readonly Func<LayaQuestionType, int, double> _temperature;

        /// <param name="onnxModelPath">Path to the exported Laya ONNX graph.</param>
        /// <param name="tokenizerJsonPath">Path to that checkpoint's tokenizer.json.</param>
        /// <param name="bosToken">Override for the "start" marker token (auto-detects "[CLS]"/"&lt;bos&gt;"/"&lt;s&gt;" otherwise).</param>
        /// <param name="eosToken">Override for the "separator" token (auto-detects "[SEP]"/"&lt;eos&gt;"/"&lt;/s&gt;" otherwise).</param>
        /// <param name="maskToken">Override for the per-option marker token (auto-detects "[MASK]"/"&lt;mask&gt;" otherwise).</param>
        /// <param name="maxLen">Total token budget for the sequence (Laya's default is 512).</param>
        /// <param name="optionMaxTokens">Max tokens kept per option label (Laya's default is 48).</param>
        /// <param name="temperature">Optional calibration temperature as a function of (question type, option count).
        /// Defaults to 1.0 (no calibration) - pass the real values from a checkpoint's rl_agent_config.json for
        /// properly calibrated probabilities.</param>
        /// <param name="preTokenizer">Override for the tokenizer's pre-tokenizer, needed for non-English checkpoints.</param>
        /// <param name="sessionOptions">Optional ONNX Runtime session options (execution provider, thread count, etc).</param>
        public LayaModel(
            string onnxModelPath,
            string tokenizerJsonPath,
            string? bosToken = null,
            string? eosToken = null,
            string? maskToken = null,
            int maxLen = 512,
            int optionMaxTokens = 48,
            Func<LayaQuestionType, int, double>? temperature = null,
            PreTokenizer? preTokenizer = null,
            SessionOptions? sessionOptions = null)
        {
            _session = sessionOptions is null
                ? new InferenceSession(onnxModelPath)
                : new InferenceSession(onnxModelPath, sessionOptions);

            var loaded = HfBpeLoader.Load(tokenizerJsonPath, preTokenizer);
            _tokenizer = loaded.Tokenizer;
            _maxLen = maxLen;
            _optionMaxTokens = optionMaxTokens;
            _temperature = temperature ?? ((_, __) => 1.0);

            _bosId = ResolveSpecialToken(loaded.TokenToId, bosToken, "[CLS]", "<bos>", "<s>");
            _eosId = ResolveSpecialToken(loaded.TokenToId, eosToken, "[SEP]", "<eos>", "</s>");
            _maskId = ResolveSpecialToken(loaded.TokenToId, maskToken, "[MASK]", "<mask>");
        }

        /// <summary>Pick the best of several plain-string options (key and description are the same).</summary>
        public LayaAnswer AskChoice(string state, string instructions, IReadOnlyList<string> options)
            => Ask(LayaQuestionType.Choice, state, instructions, ToOptions(options));

        /// <summary>Pick the best of several labelled options, given as a criteria dictionary (key -> description),
        /// e.g. <c>{ "reship": "Send a replacement", "notify": "Notify the customer of the delay" }</c>. The answer's
        /// <see cref="LayaAnswer.Options"/> will contain the keys ("reship"), not the descriptions.</summary>
        public LayaAnswer AskChoice(string state, string instructions, IReadOnlyDictionary<string, string> criteria)
            => Ask(LayaQuestionType.Choice, state, instructions, ToOptions(criteria));

        /// <summary>Score the state against an ordered list of plain-string options (e.g. a 1-5 severity scale).
        /// List/array order is always preserved, so this is safe as-is.</summary>
        public LayaAnswer AskScore(string state, string instructions, IReadOnlyList<string> options)
            => Ask(LayaQuestionType.Score, state, instructions, ToOptions(options));

        /// <summary>Score the state against an ordered criteria list (key -> description) - the order defines the
        /// scale, so this takes a list of pairs rather than a <c>Dictionary</c>/<c>IReadOnlyDictionary</c>, whose
        /// enumeration order isn't part of their contract. E.g.
        /// <c>new List&lt;KeyValuePair&lt;string,string&gt;&gt; { new("low", "..."), new("high", "...") }</c>.</summary>
        public LayaAnswer AskScore(string state, string instructions, IReadOnlyList<KeyValuePair<string, string>> criteria)
            => Ask(LayaQuestionType.Score, state, instructions, ToOptions(criteria));

        /// <summary>Ask whether a proposition holds for the state. Returns Probabilities[1] as P(true).</summary>
        public LayaAnswer AskYesNo(string state, string instructions, string noLabel = "no", string yesLabel = "yes")
            => Ask(LayaQuestionType.YesNo, state, instructions, new List<LayaOption> { new LayaOption(noLabel), new LayaOption(yesLabel) });

        private static List<LayaOption> ToOptions(IReadOnlyList<string> options)
            => options.Select(o => new LayaOption(o, o)).ToList();

        private static List<LayaOption> ToOptions(IReadOnlyDictionary<string, string> criteria)
            => criteria.Select(kvp => new LayaOption(kvp.Key, kvp.Value)).ToList();

        // Same conversion, but IReadOnlyList<KeyValuePair<...>> is what actually guarantees enumeration happens in
        // list order - a plain Dictionary/IReadOnlyDictionary doesn't promise that as part of its contract.
        private static List<LayaOption> ToOptions(IReadOnlyList<KeyValuePair<string, string>> criteria)
            => criteria.Select(kvp => new LayaOption(kvp.Key, kvp.Value)).ToList();

        /// <summary>The general entry point behind AskChoice/AskScore/AskYesNo, in case you want to pick the
        /// question type and option labels yourself.</summary>
        public LayaAnswer Ask(LayaQuestionType questionType, string state, string instructions, IReadOnlyList<LayaOption> options)
        {
            if (options.Count == 0)
            {
                throw new ArgumentException("At least one option is required.", nameof(options));
            }

            string typeWord = questionType switch
            {
                LayaQuestionType.Choice => "choice",
                LayaQuestionType.Score => "score",
                LayaQuestionType.YesNo => "noul",
                _ => throw new ArgumentOutOfRangeException(nameof(questionType)),
            };

            // Sequence layout (Laya's SDK / ONNX export contract):
            // [BOS] <type> question: <instructions> [EOS] ([MASK] option)* [EOS] <state> [EOS]
            var seq = new List<long> { _bosId };
            seq.AddRange(_tokenizer.EncodeToIds($"{typeWord} question: {instructions}").Select(id => (long)id));
            seq.Add(_eosId);

            var markerPositions = new List<long>(options.Count);
            foreach (var option in options)
            {
                markerPositions.Add(seq.Count); // position of the [MASK] we're about to add
                seq.Add(_maskId);
                var optionIds = _tokenizer.EncodeToIds(" " + option.Description);
                seq.AddRange(optionIds.Take(_optionMaxTokens).Select(id => (long)id));
            }
            seq.Add(_eosId);

            int remainingBudget = _maxLen - 1 - seq.Count; // reserve one slot for the final terminator
            if (remainingBudget > 0)
            {
                var stateIds = _tokenizer.EncodeToIds(state);
                seq.AddRange(stateIds.Take(remainingBudget).Select(id => (long)id));
            }
            seq.Add(_eosId);

            int seqLen = seq.Count;
            int k = options.Count;

            var inputIds = new DenseTensor<long>(seq.ToArray(), new[] { 1, seqLen });
            var attentionMask = new DenseTensor<long>(Enumerable.Repeat(1L, seqLen).ToArray(), new[] { 1, seqLen });
            var markerPos = new DenseTensor<long>(markerPositions.ToArray(), new[] { 1, k });
            var markerMask = new DenseTensor<bool>(Enumerable.Repeat(true, k).ToArray(), new[] { 1, k });
            var qtype = new DenseTensor<long>(new[] { (long)(int)questionType }, new[] { 1 });

            using var results = _session.Run(new[]
            {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor("marker_pos", markerPos),
            NamedOnnxValue.CreateFromTensor("marker_mask", markerMask),
            NamedOnnxValue.CreateFromTensor("qtype", qtype),
        });

            var logits = results.First(r => r.Name == "logits").AsTensor<float>();

            var scaled = new double[k];
            double temp = _temperature(questionType, k);
            for (int i = 0; i < k; i++)
            {
                scaled[i] = logits[0, i] / temp;
            }

            var probabilities = Softmax(scaled);
            int bestIndex = 0;
            for (int i = 1; i < k; i++)
            {
                if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
            }

            double expectedScore = 0;
            for (int i = 0; i < k; i++)
            {
                expectedScore += i * probabilities[i];
            }

            return new LayaAnswer
            {
                Options = options.Select(o => o.Key).ToArray(),
                Descriptions = options.Select(o => o.Description).ToArray(),
                Probabilities = probabilities,
                BestIndex = bestIndex,
                ExpectedScore = expectedScore,
            };
        }

        private static long ResolveSpecialToken(IReadOnlyDictionary<string, int> tokenToId, string? explicitToken, params string[] fallbackCandidates)
        {
            if (explicitToken != null)
            {
                if (!tokenToId.TryGetValue(explicitToken, out int id))
                {
                    throw new ArgumentException($"Token '{explicitToken}' was not found in the tokenizer vocabulary.");
                }
                return id;
            }

            foreach (var candidate in fallbackCandidates)
            {
                if (tokenToId.TryGetValue(candidate, out int id))
                {
                    return id;
                }
            }

            throw new InvalidOperationException(
                $"Could not find any of [{string.Join(", ", fallbackCandidates)}] in the tokenizer vocabulary. " +
                "Pass the correct token explicitly (bosToken/eosToken/maskToken) to the LayaModel constructor.");
        }

        private static double[] Softmax(double[] logits)
        {
            double max = logits.Max();
            var exp = logits.Select(l => Math.Exp(l - max)).ToArray();
            double sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }

        public void Dispose() => _session.Dispose();
    }

}