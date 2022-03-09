import math
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable

import __init_paths
#from face_model import model
from face_model.gpen_model import FullGenerator

def model_load(model, path):
    """Load model."""

    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    model.load_state_dict(state_dict)


def export_onnx(model, path, force_cpu):
    """Export onnx model."""

    import onnx
    import onnxruntime
    #from onnx import optimizer
    import numpy as np

    onnx_file_name = os.path.join(path, model+".onnx")
    model_weight_file = os.path.join(path, model+".pth")
    dummy_input = Variable(torch.randn(1, 3, 1024, 1024))

    # 1. Create and load model.
    model_setenv(force_cpu)
    torch_model = get_model(model_weight_file)
    torch_model.eval()

    # 2. Model export
    print("Export model ...")

    input_names = ["input"]
    output_names = ["output"]
    device = model_device()
    # torch.onnx.export(torch_model, dummy_input.to(device), onnx_file_name,
                  # input_names=input_names,
                  # output_names=output_names,
                  # verbose=False,
                  # opset_version=12,
                  # keep_initializers_as_inputs=False,
                  # export_params=True,
                  # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    torch.onnx.export(torch_model, dummy_input.to(device), onnx_file_name,
                  input_names=input_names,
                  output_names=output_names,
                  verbose=False,
                  opset_version=10,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    # 3. Optimize model
    print('Checking model ...')
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)
    # https://github.com/onnx/optimizer
    print('Done checking model ...')
    # 4. Visual model
    # python -c "import netron; netron.start('output/image_zoom.onnx')"

def verify_onnx(model, path, force_cpu):
    """Verify onnx model."""

    import onnxruntime
    import numpy as np


    model_weight_file = os.path.join(path, model+".pth")

    model_weight_file = "./weights/GPEN-512.pth"

    model_setenv(force_cpu)
    torch_model = get_model(model_weight_file)
    torch_model.eval()

    onnx_file_name = os.path.join(path, model+".onnx")
    onnxruntime_engine = onnxruntime.InferenceSession(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = Variable(torch.randn(1, 3, 512, 512))
    with torch.no_grad():
        torch_output, _ = torch_model(dummy_input)
    onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example: Onnx model has been tested with ONNXRuntime, the result looks good !")

def get_model(checkpoint):
    """Create encoder model."""

    #model_setenv()
    model = FullGenerator(1024, 512, 8, 2, narrow=1) #TODO
    model_load(model, checkpoint)
    device = model_device()
    model.to(device)
    return model


def model_device():
    """Please call after model_setenv. """

    return torch.device(os.environ["DEVICE"])


def model_setenv(cpu_only):
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if cpu_only:
        os.environ["DEVICE"] = 'cpu'
    else:
        if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
            os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.environ["DEVICE"] == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    #print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])

# def export_torch(model, path):
    # """Export torch model."""

    # script_file = os.path.join(path, model+".pt")
    # weight_file = os.path.join(path, model+".onnx")

    # # 1. Load model
    # print("Loading model ...")
    # model = get_model(weight_file)
    # model.eval()

    # # 2. Model export
    # print("Export model ...")
    # dummy_input = Variable(torch.randn(1, 3, 512, 512))
    # device = model_device()
    # traced_script_module = torch.jit.trace(model, dummy_input.to(device), _force_outplace=True)
    # traced_script_module.save(script_file)


if __name__ == '__main__':
    """Test model ..."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--export', help="Export onnx model", action='store_true')
    parser.add_argument('--verify', help="Verify onnx model", action='store_true')
    parser.add_argument('--force-cpu', dest='force_cpu', help="Verify onnx model", action='store_true')

    args = parser.parse_args()

    # export_torch()
    
    

    if args.export:
        export_onnx(model = args.model, path = args.path, force_cpu=args.force_cpu)

    if args.verify:
        verify_onnx(model = args.model, path = args.path, force_cpu=args.force_cpu)
