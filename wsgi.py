from application import app
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

if __name__ == '__main__':
    # serve(app,host='0.0.0.0')
    app.run(host='0.0.0.0')
