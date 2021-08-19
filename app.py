from flask import Flask, request, render_template_string
import coremltools as ct
from PIL import Image

app = Flask(__name__)
model = ct.models.MLModel('FaceID.mlmodel')

@app.route('/', methods=['POST'])
def index():
	if request.files:
		img = Image.open(request.files['image'])
		out_dict = model.predict({'image': img})

		if out_dict["classLabelProbs"]['Ethan'] > 0.95:
			return "Success"
		else:
			return "Failure"
	else:
		return "Invalid image format"

if __name__ == '__main__':
	app.run(threaded=True, port=5000)