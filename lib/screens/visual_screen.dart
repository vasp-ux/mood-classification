import 'package:flutter/material.dart';

class VisualScreen extends StatefulWidget {
  const VisualScreen({super.key});

  @override
  State<VisualScreen> createState() => _VisualScreenState();
}

class _VisualScreenState extends State<VisualScreen> {
  bool isRunning = false;
  DateTime? startTime;
  Duration sessionDuration = Duration.zero;

  void startSession() {
    setState(() {
      isRunning = true;
      startTime = DateTime.now();
    });
  }

  void stopSession() {
    setState(() {
      isRunning = false;
      sessionDuration = DateTime.now().difference(startTime!);
    });

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => VisualResultScreen(
          duration: sessionDuration,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Visual Mood Detection"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              height: 220,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.black12,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Center(
                child: Icon(
                  Icons.camera_alt,
                  size: 80,
                  color: isRunning ? Colors.red : Colors.grey,
                ),
              ),
            ),
            const SizedBox(height: 30),

            Text(
              isRunning ? "Camera Running..." : "Camera Stopped",
              style: const TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 20),

            isRunning
                ? ElevatedButton.icon(
                    onPressed: stopSession,
                    icon: const Icon(Icons.stop),
                    label: const Text("Stop Camera"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      padding: const EdgeInsets.all(14),
                    ),
                  )
                : ElevatedButton.icon(
  onPressed: startSession,
  icon: const Icon(Icons.play_arrow),
  label: const Text("Start Camera"),
  style: ElevatedButton.styleFrom(
    padding: const EdgeInsets.all(14),
  ),
),


          ],
        ),
      ),
    );
  }
}


class VisualResultScreen extends StatelessWidget {
  final Duration duration;

  const VisualResultScreen({super.key, required this.duration});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Session Result"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "Session Duration: ${duration.inSeconds} seconds",
              style: const TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 20),

            const Text(
              "Detected Mood (Demo):",
              style: TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 10),

            const Text(
              "🧠 Neutral",
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.indigo,
              ),
            ),

            const SizedBox(height: 30),

            const Text(
              "Emotion Distribution (Sample):",
              style: TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 10),

            const Text("Happy: 30%"),
            const Text("Neutral: 50%"),
            const Text("Surprise: 20%"),
          ],
        ),
      ),
    );
  }
}
