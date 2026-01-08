import 'package:flutter/material.dart';

class SummaryScreen extends StatefulWidget {
  const SummaryScreen({super.key});

  @override
  State<SummaryScreen> createState() => _SummaryScreenState();
}

class _SummaryScreenState extends State<SummaryScreen> {
  String selectedPeriod = "Today";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Mood Summary"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Select Time Period",
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 10),

            DropdownButtonFormField<String>(
              value: selectedPeriod,
              items: const [
                DropdownMenuItem(value: "Today", child: Text("Today")),
                DropdownMenuItem(value: "Last 7 Days", child: Text("Last 7 Days")),
              ],
              onChanged: (value) {
                setState(() {
                  selectedPeriod = value!;
                });
              },
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 30),

            ElevatedButton(
              onPressed: () {},
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(14),
              ),
              child: const Text("View Summary"),
            ),

            const SizedBox(height: 30),

            const Divider(),

            const SizedBox(height: 20),

            const Text(
              "Text Mood:",
              style: TextStyle(fontSize: 18),
            ),
            const Text(
              "Happy (Demo)",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),

            const SizedBox(height: 20),

            const Text(
              "Visual Mood:",
              style: TextStyle(fontSize: 18),
            ),
            const Text(
              "Neutral (Demo)",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),

            const SizedBox(height: 30),

            const Text(
              "🧠 Final Fused Mood:",
              style: TextStyle(fontSize: 20),
            ),
            const SizedBox(height: 10),

            const Text(
              "Neutral",
              style: TextStyle(
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
