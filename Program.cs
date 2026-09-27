using LayaOnnx;

if (args.Length < 2)
{
    Console.WriteLine("Usage: LayaOnnx <path-to-model.onnx> <path-to-tokenizer.json>");
    return;
}

string onnxPath = args[0];
string tokenizerPath = args[1];

using var laya = new LayaModel(onnxPath, tokenizerPath);

// Example 1: choice
var intent = laya.AskChoice(
    state: "Hi, my order #48213 never arrived and it's been two weeks. I'd like my money back.",
    instructions: "What does the customer want?",
    options: new[] { "refund", "replacement", "order status update", "general complaint" });

Console.WriteLine($"Intent: {intent.BestOption} ({intent.Confidence:P1})");
for (int i = 0; i < intent.Options.Length; i++)
{
    Console.WriteLine($"  {intent.Options[i]}: {intent.Probabilities[i]:P1}");
}

// Example 2: yes/no
var isUrgent = laya.AskYesNo(
    state: "Hi, my order #48213 never arrived and it's been two weeks. I'd like my money back.",
    instructions: "Is the customer angry or frustrated?");

Console.WriteLine($"\nFrustrated? {isUrgent.BestOption} (P(true) = {isUrgent.Probabilities[1]:P1})");

// Example 3: score
var severity = laya.AskScore(
    state: "Hi, my order #48213 never arrived and it's been two weeks. I'd like my money back.",
    instructions: "How severe is this support ticket?",
    options: new[] { "low", "medium", "high", "critical" });

Console.WriteLine($"\nSeverity: {severity.BestOption} (expected index {severity.ExpectedScore:F2})");
