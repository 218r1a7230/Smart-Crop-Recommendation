🌱 Smart Crop Recommendation System

📌 Overview
The Smart Crop Recommendation System uses Machine Learning to predict the most suitable crop for a given set of soil and climate parameters. It analyzes inputs such as Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall to recommend the best crop.

🚀 Features
- Uses Random Forest, SVM, Naive Bayes, MLP, and Gradient Boosting models.
- Implements a Hybrid Model (Voting Classifier) for better accuracy.
- Displays Graphical Insights (Feature Distribution, Confusion Matrix, ROC Curve, Model Comparison).
- User-Friendly Web Interface built with Flask.
- Interactive Step-by-Step Form to enter soil and climate data.

--------------------------------------------------------

🛠️ Installation & Setup
1️⃣ Clone the Repository
    git clone https://github.com/yourusername/Smart-Crop-Recommendation.git
    cd Smart-Crop-Recommendation

2️⃣ Create a Virtual Environment (Recommended)
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate  # For Windows

3️⃣ Install Dependencies
    pip install -r requirements.txt

--------------------------------------------------------

🏃‍♂️ Run the Application
    python app.py
Then open http://127.0.0.1:5000/ in your browser.

--------------------------------------------------------

📊 Graphical Insights
The system generates insightful plots stored in static/plots/:
- Feature Distribution
- Correlation Heatmap
- Confusion Matrix
- Precision-Recall Curve
- ROC Curve
- Model Comparison (Individual vs Hybrid)

--------------------------------------------------------

🔬 How It Works
1. User Inputs Data (Soil & Climate conditions).
2. Machine Learning Model Predicts the Best Crop.
3. Accuracy & Graphs are Displayed to explain model performance.

--------------------------------------------------------

🏗️ Project Structure
project/
│
├── app.py               # Flask app (Main API)
├── model.py             # ML Model Training & Prediction
├── requirements.txt     # Required dependencies
│
├── templates/
│   ├── index.html       # Main UI page
│   ├── form_fields.html # Step-by-step form input
│
├── static/
│   ├── styles.css       # CSS Styling
│   ├── scripts.js       # JavaScript for UI interaction
│   ├── plots/           # Folder where graphs are saved
│
└── saved_models/        # Trained Machine Learning models

--------------------------------------------------------

🎯 Technologies Used
- Python, Flask (Backend)
- Machine Learning (Scikit-Learn)
- Bootstrap, JavaScript, HTML, CSS (Frontend)
- Matplotlib, Seaborn (Data Visualization)

--------------------------------------------------------

🤝 Contributing
If you'd like to contribute:
1. Fork this repository.
2. Create a feature branch:
    git checkout -b feature-branch
3. Make changes & commit:
    git commit -m "Added new feature"
4. Push & create a Pull Request.

--------------------------------------------------------

📜 License
This project is open-source and available under the MIT License.

--------------------------------------------------------

📞 Contact
👨‍💻 Developer: Praneeth Kalyan Gurramolla
📧 Email: 218r1a7230.cmrec@gmail.com
📌 GitHub: https://github.com/218r1a7230

--------------------------------------------------------

🚀 Enjoy using the Smart Crop Recommendation System! 🌾✨
