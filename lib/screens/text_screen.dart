import 'package:flutter/material.dart';
import '../services/api_service.dart';


class TextScreen extends StatefulWidget {
  const TextScreen({super.key});

  @override
  State<TextScreen> createState() => _TextScreenState();
}

class _TextScreenState extends State<TextScreen> {
  final TextEditingController textController = TextEditingController();
  bool isAnalyzing = false;

  void analyzeText() async {
  if (textController.text.trim().isEmpty) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Please enter some text")),
    );
    return;
  }

  setState(() {
    isAnalyzing = true;
  });

  try {
    final emotion =
        await ApiService.predictText(textController.text);

    print("BACKEND EMOTION = $emotion"); // 🔍 debug

    setState(() {
      isAnalyzing = false;
    });

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => TextResultScreen(
          inputText: textController.text,
          emotion: emotion, // 🔥 REAL VALUE
        ),
      ),
    );
  } catch (e) {
    setState(() {
      isAnalyzing = false;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Error contacting backend")),
    );
  }
}


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Text Mood Detection"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            const Text(
              "Enter your thoughts / diary text below:",
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 15),

            TextField(
              controller: textController,
              maxLines: 6,
              decoration: const InputDecoration(
                hintText: "Type here...",
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 20),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: isAnalyzing ? null : analyzeText,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(14),
                ),
                child: isAnalyzing
                    ? const CircularProgressIndicator(
                        color: Colors.white,
                      )
                    : const Text("Analyze Text"),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class TextResultScreen extends StatelessWidget {
  final String inputText;
  final String emotion;

  const TextResultScreen({
    super.key,
    required this.inputText,
    required this.emotion,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Text Session Result"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Your Input:",
              style: TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 8),

            Text(
              inputText,
              style: const TextStyle(fontStyle: FontStyle.italic),
            ),

            const SizedBox(height: 30),

            const Text(
              "Detected Mood:",
              style: TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 10),

            Text(
              "🧠 $emotion",
              style: const TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.indigo,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
