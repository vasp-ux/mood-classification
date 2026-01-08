import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://127.0.0.1:5000";

  static Future<String> predictText(String text) async {
    final response = await http.post(
      Uri.parse("$baseUrl/predict/text"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"text": text}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data["emotion"];
    } else {
      throw Exception("Failed to predict emotion");
    }
  }
}
