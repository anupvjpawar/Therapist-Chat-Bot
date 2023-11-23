BlissBot

BlissBot is a therapeutic chatbot designed to provide a supportive and comforting virtual environment. This project combines natural language processing with image recognition to create an interactive experience for users. BlissBot evaluates user inputs and provides predictions about the emotional content of the text, accompanied by visual cues.

Features


Text Analysis: Utilizes natural language processing techniques to analyze user-provided text.
Emotional Prediction: Predicts whether the input text is flagged as concerning content or considered fine.
Interactive Interface: Integrates with a user-friendly web interface powered by Dash, allowing users to interact with BlissBot easily.
Visual Feedback: Enhances user experience with visual feedback using OpenCV to display relevant images based on predictions.
Getting Started
Clone the repository:

bash
Copy code
git clone https://github.com/anupvjpawar/blissbot.git
cd blissbot
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the BlissBot app:

bash
Copy code
python blissbot.py
Open your browser and navigate to http://localhost:8053 to interact with BlissBot.

![output1](https://github.com/anupvjpawar/Therapist-Chat-Bot/assets/76862648/946a9057-da97-470b-86fb-3501679e2d24)


![output2](https://github.com/anupvjpawar/Therapist-Chat-Bot/assets/76862648/9848e493-5a95-4b3e-abdc-037614721d93)


Project Structure
blissbot.py: The main Dash web application script.
Sheet_1.csv: Dataset containing text responses used for training the model.
url_to_not_flagged_image.jpg: Image representing a non-flagged scenario.
url_to_flagged_image.jpg: Image representing a flagged scenario.
backgroundimg.jpg: Background image for the Dash web interface.
Contributing
Contributions are welcome! Whether you're a developer, designer, or enthusiast, feel free to contribute to make BlissBot even more delightful.
