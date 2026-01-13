using System;
using System.Threading.Tasks;
using DotNetEnv;
using OpenAI;
using OpenAI.Chat;

namespace Week1AIApp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Simple C# AI Console App");
            Console.WriteLine("Calling OpenAI once, then exiting.\n");

            // Load .env
            Env.Load();

            // Get API key from environment
            var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                Console.WriteLine("OPENAI_API_KEY is not set. Check your .env file.");
                return;
            }

            // Create OpenAI client
            var client = new OpenAIClient(apiKey);

            // Build a simple chat request, similar to your Python prompt
            var chatClient = client.GetChatClient("gpt-4.1-mini"); // or another model you have access to

            var result = await chatClient.CompleteChatAsync(
                "What is the difference between supervised and unsupervised training in ML?"
            );

            Console.WriteLine("AI response:\n");
            Console.WriteLine(result);

            Console.WriteLine("\nDone.");
        }
    }
}
