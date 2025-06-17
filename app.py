from flask import Flask, request, render_template
from corporate_search_agent import search_corporate_agent
import json

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        result = search_corporate_agent(query)
        return render_template('index.html', query=query, result=json.dumps(result, ensure_ascii=False, indent=2))
    return render_template('index.html', query='', result='')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
