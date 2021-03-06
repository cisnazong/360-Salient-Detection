from flask import Flask, render_template, redirect, url_for, Response
import os

app = Flask(__name__)
resources_root = './resources'
video_path = 'video'


def generate_video_bitstream(filename):
    f = open(os.path.join(resources_root, video_path, filename), mode='rb')
    while True:
        data_b = f.read(40960)
        if data_b == b"":
            break
        yield data_b
    f.close()


@app.route('/video/<filename>')
def get_video(filename):
    return Response(generate_video_bitstream(filename))


@app.route('/display/<filename>')
def show_player(filename=None):
    return render_template('display.html', filename=filename)


@app.route('/display2/<filename>')
def show_player2(filename=None):
    return render_template('display2.html', filename=filename)


@app.route('/display_multi/<video_list_str>')
def show_player_multi(video_list_str=None):
    print('[+]' + ' parameters: ' + video_list_str)
    video_list = video_list_str.split('&')
    print('[+]' + ' streaming videos: ' + str(video_list))
    return render_template('display_multi.html', video_list=video_list)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print('[+]'+' current working directory: ' + os.getcwd())
    app.run(threaded=True)
