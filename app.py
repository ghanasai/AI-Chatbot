from flask import Flask, request, render_template
from markupsafe import Markup
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'C:/Users/91903/Desktop/SITCollegeEnquiryAIChatBot/src/'))
from chat_module import Chatbot


app = Flask(__name__, static_folder='static')
chat_history = []
chatbot = Chatbot()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["message"]
        bot_response = chatbot.start_chatbot(user_input)
        chat_history.append((user_input,Markup(bot_response)))
    return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)