import 'package:flutter/material.dart';
import 'visual_screen.dart';
import 'text_screen.dart';
import 'profile_screen.dart';
import 'summary_screen.dart';



class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Dashboard"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.count(
          crossAxisCount: 2,
          crossAxisSpacing: 16,
          mainAxisSpacing: 16,
          children: [
            _dashboardCard(
  context,
  icon: Icons.camera_alt,
  label: "Visual Input",
  onTap: () {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const VisualScreen(),
      ),
    );
  },
),

            _dashboardCard(
              context,
              icon: Icons.edit,
              label: "Text Input",
              onTap: () {
  Navigator.push(
    context,
    MaterialPageRoute(
      builder: (context) => const TextScreen(),
    ),
  );
},

            ),
            _dashboardCard(
              context,
              icon: Icons.analytics,
              label: "Summary",
             onTap: () {
  Navigator.push(
    context,
    MaterialPageRoute(
      builder: (context) => const SummaryScreen(),
    ),
  );
},

            ),
            _dashboardCard(
              context,
              icon: Icons.person,
              label: "Profile",
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const ProfileScreen(),

                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _dashboardCard(
    BuildContext context, {
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.indigo.shade50,
          borderRadius: BorderRadius.circular(16),
          boxShadow: const [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 6,
              offset: Offset(0, 4),
            )
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 48, color: Colors.indigo),
            const SizedBox(height: 12),
            Text(
              label,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            )
          ],
        ),
      ),
    );
  }
}

class PlaceholderScreen extends StatelessWidget {
  final String title;

  const PlaceholderScreen({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: Center(
        child: Text(
          "$title Screen\n(Will be implemented next)",
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 20),
        ),
      ),
    );
  }
}
