import 'package:flutter/material.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final _formKey = GlobalKey<FormState>();

  final nameController = TextEditingController();
  final ageController = TextEditingController();
  final courseController = TextEditingController();

  String gender = "Male";
  double completion = 0.0;

  void calculateCompletion() {
    int filled = 0;
    if (nameController.text.isNotEmpty) filled++;
    if (ageController.text.isNotEmpty) filled++;
    if (courseController.text.isNotEmpty) filled++;
    if (gender.isNotEmpty) filled++;

    setState(() {
      completion = filled / 4;
    });
  }

  void saveProfile() {
    if (_formKey.currentState!.validate()) {
      calculateCompletion();

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Profile saved successfully")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Profile Completion"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              const Text(
                "Complete your profile",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 15),

              LinearProgressIndicator(
                value: completion,
                minHeight: 8,
              ),
              const SizedBox(height: 20),

              TextFormField(
                controller: nameController,
                decoration: const InputDecoration(
                  labelText: "Name",
                  border: OutlineInputBorder(),
                ),
                onChanged: (_) => calculateCompletion(),
                validator: (value) =>
                    value!.isEmpty ? "Enter name" : null,
              ),
              const SizedBox(height: 15),

              TextFormField(
                controller: ageController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: "Age",
                  border: OutlineInputBorder(),
                ),
                onChanged: (_) => calculateCompletion(),
                validator: (value) =>
                    value!.isEmpty ? "Enter age" : null,
              ),
              const SizedBox(height: 15),

              DropdownButtonFormField<String>(
                value: gender,
                decoration: const InputDecoration(
                  labelText: "Gender",
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(value: "Male", child: Text("Male")),
                  DropdownMenuItem(value: "Female", child: Text("Female")),
                  DropdownMenuItem(value: "Other", child: Text("Other")),
                ],
                onChanged: (value) {
                  gender = value!;
                  calculateCompletion();
                },
              ),
              const SizedBox(height: 15),

              TextFormField(
                controller: courseController,
                decoration: const InputDecoration(
                  labelText: "Course / Profession",
                  border: OutlineInputBorder(),
                ),
                onChanged: (_) => calculateCompletion(),
                validator: (value) =>
                    value!.isEmpty ? "Enter course or profession" : null,
              ),
              const SizedBox(height: 30),

              ElevatedButton(
                onPressed: saveProfile,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(14),
                ),
                child: const Text("Save Profile"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
