# -*- coding: utf-8 -*-
"""AIbro Inference Demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NH2Aj1bbCgqJXyNKK9TkzwlO4JDBkTQC

# **Welcome to AIbro Inference Demo!**
In this demo, we will show how you can deploy an AI model in 2 minutes. All you need is a formatted ML model repo and an ML application scenario.

<img src="https://drive.google.com/uc?export=view&id=1Tp4w9bd3Yf3_e1gf1_CdY5aNwZ48mwvm" width="600" height="500" />

## Step 1: Install AIbro
"""

!pip install aibro
!sudo apt-get -o Dpkg::Options::="--force-confmiss" install --reinstall netbase # this command is only needed if you meet error: "OSError: protocol not found". Colab is in this case.
!apt-get install python3.7-dev python3.7-venv # this command is only need for Colab

"""## Step 2: Prepare a formatted model repo

Source: [https://github.com/AIpaca-Inc/Aibro_examples](https://github.com/AIpaca-Inc/Aibro-examples).

The repo should be structured in the following format:

repo <br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_&nbsp;[predict.py](#predict-py)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_&nbsp;[model](#39-model-39-and-39-data-39-folders)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_&nbsp;[data](#39-model-39-and-39-data-39-folders)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_&nbsp;[requirement.txt](#requirement-txt)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;|\_\_&nbsp;[other artifacts](#other-artifacts)<br/>

### **predict.py**

This is the entry point of AIbro.

predict.py should contain two methods:

1. _load_model()_: this method should load and return your machine learning model from the "model" folder. An transformer-based Portuguese to English translator is used in this example repo.

```python
def load_model():
    # Portuguese to English translator
    translator = tf.saved_model.load('model')
    return translator
```

2. _run()_: this method used model as the input, load data from the "data" folder, predict, then return the inference result.

```python
def run(model):
    fp = open("./data/data.json", "r")
    data = json.load(fp)
    sentence = data["data"]

    result = {"data": model(sentence).numpy().decode("utf-8")}
    return result
```

**test tip**: predict.py() should be able to return an inference result by:

```python
run(load_model())
```

### **"model" and "data" folders**

There is no format restriction on the "model" and "data" folder as long as the input and output of load_model() and run() from predict.py are correct.

### **requirement.txt**

Before start deploying the model, packages from requirement.txt are installed to setup the environment.

### **Other Artifacts**

All other files/folders.

"""

!git clone https://github.com/napo178/abicai.git

"""## Step 3: Test the Repo by Dryrun

Dryrun locally validates the repo structure and tests if inference result can be successfully returned.
"""

from aibro.inference import Inference
Inference.deploy(
    "./Aibro-examples/tensorflow_transformer",
    dryrun=True,
)

"""## Step 4: Create an inference API with one-line code
Assume the formatted model repo is saved at path "./aibro_repo", we can now use it to create an inference job. The model name should be unique respect to all current [active inference jobs](https://aipaca.ai/inference_jobs) under your profile.

In this example, we deployed a public custom model from "./aibro_repo" called "my_fancy_transformer" on machine type "c5.large.od" and used access token for authentication.

Once the deployment finished, an API URL is returned with the syntax: </br>

- **https://api.aipaca.ai/v1/{username}/{client_id}/{model_name}/predict** </br>

**{client_id}**: if your inference job is public, **{client_id}** is filled by "public". Otherwise, **{client_id}** should be filled by one of your [clients' ID](#add-clients).
"""

from aibro.inference import Inference

api_url = Inference.deploy(
    model_name = "my_fancy_transformer",
    machine_id_config = "c5.large.od",
    artifacts_path = "./Aibro-examples/tensorflow_transformer",
    client_ids = [] # if no clients are specified, the inference job becomes public
)

"""## Step 5: Test a Aibro API with curl
Copy your API URL into `{{api_url}}`. For instance, my `api_url` is http://api.aipaca.ai/v1/yuqil725/public/my_fancy_transformer/predict

The syntax when using `curl` depends on the file type in the `data` folder.

| Data Type | syntax                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------ |
| json      | curl -X POST {{aibro url}} -d '{"your": "data"}'<br/>curl -X POST {{aibro url}} -F file=@'path/to/json/file' |
| txt       | curl -X POST {{aibro url}} -d 'your data'<br/>curl -X POST {{aibro url}} -F file=@'path/to/txt/file'         |
| csv       | curl -X POST {{aibro url}} -F file=@'path/to/csv/file'                                                       |
| others    | curl -X POST {{aibro url}} -F file=@'path/to/zip/file'                                                       |

You may have observed some patterns from the syntax lookup table above:

- If the data type is `json` or `txt`, you could use `-d` flag to post the string data directly.
- If the data type is one of `json`, `txt`, or `csv`, you could use `-F` flag to post the data file by path.
- If the data type is not one of `json`, `txt`, or `csv`, you could zip the entire `data` folder then post the data file by the zip path.

_Tips_: if your inference time is over one minute, it is recommended to either reduce the data size or increase the `--keepalive-time` value when using `curl`.
"""

!curl -X POST {{api_url}} -d '{"data": "Olá"}'

"""## Step 6: Limit API Access to Specific Clients (Optional)

As the API owner, you probably don't receive overwhelming API requests from everywhere. To avoid this trouble, you could give every client an unique client id, which is going to used in API endpoint (as the shown syntax in the step 4). If no client id was added, this inference job would be public by default.
"""

from aibro.inference import Inference
Inference.update_clients(
    job_id = "inf_ec49d03f-67ba-44e8-ac3c-c6bc81ca630c",
    add_client_ids = ["client_1", "client_2"]
)

"""### Run a Private Prediction

You could fill in {client_id} by either "client_1" or "client_2" now. "public" is not going to work any more.
"""

!curl -d '{"data": "Olá"}' -X POST {{api_url}}

!curl -d '{"data": "Olá"}' -X POST {{api_url}}

"""## Step 7: Complete Job

Once the inference job is no longer used, to avoid unnecessary cost, please remember to close it by `Inference.complete()`.
"""

Inference.complete(job_id="inf_ec49d03f-67ba-44e8-ac3c-c6bc81ca630c")
