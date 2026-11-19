using System.Text;
using System.Text.RegularExpressions;

namespace RagInABox.Services;

/// <summary>
/// Splits text into chunks of at most <paramref name="maxChars"/> characters,
/// packing whole sentences together and carrying a small overlap from the
/// previous chunk so an idea split across a boundary is not lost.
/// </summary>
public static class Chunker
{
    public static List<string> Chunk(string text, int maxChars = 1000, int overlapChars = 150)
    {
        var chunks = new List<string>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return chunks;
        }

        var sentences = Regex.Split(text, @"(?<=[.!?])\s+")
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToList();

        var current = new StringBuilder();

        foreach (var sentence in sentences)
        {
            if (current.Length > 0 && current.Length + sentence.Length + 1 > maxChars)
            {
                chunks.Add(current.ToString());
                var tail = current.ToString();
                current.Clear();

                if (overlapChars > 0 && tail.Length > overlapChars)
                {
                    current.Append(tail[^overlapChars..]).Append(' ');
                }
            }

            if (current.Length > 0)
            {
                current.Append(' ');
            }

            current.Append(sentence);
        }

        if (current.Length > 0)
        {
            chunks.Add(current.ToString());
        }

        return chunks;
    }
}
