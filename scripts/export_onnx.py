import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import inspect
import json
import torch
from utils.checkpoint import load_model

def main():
    parser = argparse.ArgumentParser(description='Export STLNet to ONNX')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    if args.size < 16 or args.opset < 17:
        parser.error('size must be >=16 and opset >=17')
    import onnx
    torch.set_num_threads(4)
    model, config, _ = load_model(args.checkpoint)
    model.eval()
    x = torch.randn(1, 3, args.size, args.size)
    model.switch_to_deploy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exporter_options = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
    with torch.inference_mode():
        torch.onnx.export(model, x, str(output), input_names=['images'], output_names=['logits'], dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=args.opset, **exporter_options)
    onnx.checker.check_model(onnx.load(str(output)))
    metadata = {'input_size': args.size, 'dynamic_batch': True, 'fixed_spatial_shape': True, 'training_input_size': config['data'].get('image_size', 224), 'preprocessing': {'image_size': args.size, 'color': 'RGB', 'resize': 'bilinear', 'scale': 'uint8 / 255.0', 'mean': config['data'].get('mean', [0.5] * 3), 'std': config['data'].get('std', [0.5] * 3)}, 'output': 'one raw forgery logit per image; sigmoid for probability', 'label_mapping': {'0': 'real', '1': 'fake'}}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'onnx': str(output), **metadata}, indent=2))
if __name__ == '__main__':
    main()
