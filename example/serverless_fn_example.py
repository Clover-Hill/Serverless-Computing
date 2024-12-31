from flask import Flask
from flask import request

REQUEST_ID_HEADER = 'x-fc-request-id'

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def hello_world(path):
        rid = request.headers.get(REQUEST_ID_HEADER)
        print("FC Invoke Start RequestId: " + rid)
        data = request.stream.read()
        print("Path: " + path)
        print("Data: " + str(data))
        print("FC Invoke End RequestId: " + rid)
        return "Hello, World!"

if __name__ == '__main__':
        app.run(host='0.0.0.0',port=9000)
