using System;
using System.Collections.Generic;
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

// Example 4: choice from a criteria dictionary (key -> description) - BestOption is "reship", not the sentence
var action = laya.AskChoice(
    state: "The customer's package shows delivered but they say it never arrived, and they confirmed the address is right.",
    instructions: "What should we do next?",
    criteria: new Dictionary<string, string>
    {
        ["reship"] = "Send a replacement",
        ["notify"] = "Notify the customer of the delay",
        ["fix_address"] = "Contact the customer to confirm the address",
        ["wait"] = "Wait, no action needed",
    });

Console.WriteLine($"\nNext action: {action.BestOption} ({action.Confidence:P1})");

// Example 5: score from an ORDERED criteria list (key -> description) - order here defines the scale, so
// AskScore takes a list of KeyValuePairs rather than a plain Dictionary (whose enumeration order isn't
// guaranteed by its contract, even though it happens to preserve insertion order today).
var urgency = laya.AskScore(
    state: "Hi, my order #48213 never arrived and it's been two weeks. I'd like my money back.",
    instructions: "How urgent is this ticket, on this scale?",
    criteria: new List<KeyValuePair<string, string>>
    {
                    new KeyValuePair<string,string>("low", "Can wait a few days"),
                    new KeyValuePair<string,string>("medium", "Should be handled today"),
                    new KeyValuePair<string,string>("high", "Needs a response within the hour"),
                    new KeyValuePair<string,string>("critical", "Drop everything"),
    });

Console.WriteLine($"\nUrgency: {urgency.BestOption} (expected index {urgency.ExpectedScore:F2})");


