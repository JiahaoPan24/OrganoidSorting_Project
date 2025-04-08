<h1>REQUIREMENTS</h1>
To run the code, ensure you have the following software configuration:
Windows 10 64-bit
Python 3.9

<h1>INSTALLATION</h1>
Install all necessary packages listed in the requirements.txt file:
>pip install -r requirements.txt

<h1>RUNNING</h1>
To process images using the optimized model, use the following command in the Anaconda Prompt from the directory containing OrganoID.py:

>python OrganoID.py run OptimizedModel /path/to/images /path/to/outputFolder

This command processes each image in the specified input folder and generates labeled grayscale images saved in the specified output folder. Additional command options are available to generate different output types:
--overlay: Create detection belief images.
--track: Analysis time series images.

To view all available options and their instructions, run:

>python OrganoID.py run -h

Training the Model
If you need to customize the model for specific applications, you can re-train the included TrainableModel. To view training instructions, run:

>python OrganoID.py train -h

For example:

>python OrganoID.py train /path/to/trainingData /path/to/outputFolder NewModelName -M TrainableModel

<h1>USER INTERFACE</h1>
OrganoID also features a graphical user interface. To launch the interface, run:
>python OrganoID_UI.py

To run detection and optional classificaiton on an in-screen window:

>python OrganoID.py window OptimizedModel "Window Name" /path/to/classificaiton/model

If window name is not known or incorrect, names of open windows will be displayed during runtime. Currently only supports EfficientNetv2 Models. A classificaiton model of this type can be found at organoid_classification_non_supervised/test_model. If path is not given, only detection will be run.