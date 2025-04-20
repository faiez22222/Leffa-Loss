import numpy as np
import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Download checkpoints
snapshot_download(
    repo_id="franciszzj/Leffa",
    local_dir="./ckpts",
    allow_patterns=[
        "stable-diffusion-inpainting/*",
        "virtual_tryon.pth",
        "densepose/*",
        "schp/*",
        "humanparsing/*",
        "openpose/*",
        "examples/*",
    ],
    ignore_patterns=["pose_transfer.pth", "virtual_tryon_dc.pth"],
)

class LeffaPredictor(object):
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )
        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )
        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )
        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )
        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False
    ):
        # Open and resize the source image
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        # Handle the reference image
        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            ref_image = Image.open(ref_image_path)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            elif vt_model_type == "dress_code":
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))

        # Generate densepose
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                densepose = Image.fromarray(src_image_seg_array)

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        if control_type == "virtual_tryon" and vt_model_type == "viton_hd":
            inference = self.vt_inference_hd

        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment):
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,
        )

def run_virtual_tryon(src_image, ref_image, guidance_scale, num_steps, vt_garment_type, seed, ref_acceleration, vt_repaint, preprocess_garment):
    """
    Process uploaded images and run Leffa virtual try-on.
    
    Args:
        src_image (PIL.Image): Source image (person)
        ref_image (PIL.Image): Reference image (garment)
        guidance_scale (float): Classifier-free guidance scale
        num_steps (int): Number of inference steps
        vt_garment_type (str): Garment type (upper_body, lower_body, full_body)
        seed (int): Random seed
        ref_acceleration (bool): Enable reference acceleration
        vt_repaint (bool): Enable repaint mode
        preprocess_garment (bool): Preprocess garment image
    
    Returns:
        PIL.Image: Generated virtual try-on image
    """
    try:
        # Initialize predictor
        leffa_predictor = LeffaPredictor()
        
        # Save uploaded images temporarily
        src_image_path = "temp_src.png"
        ref_image_path = "temp_ref.png"
        src_image.save(src_image_path)
        ref_image.save(ref_image_path)
        
        # Set fixed parameters
        vt_model_type = "viton_hd"
        
        # Run inference
        gen_image, _, _ = leffa_predictor.leffa_predict_vt(
            src_image_path=src_image_path,
            ref_image_path=ref_image_path,
            ref_acceleration=ref_acceleration,
            step=num_steps,
            scale=guidance_scale,
            seed=seed,
            vt_model_type=vt_model_type,
            vt_garment_type=vt_garment_type,
            vt_repaint=vt_repaint,
            preprocess_garment=preprocess_garment
        )
        
        # Convert output to PIL Image
        return Image.fromarray(gen_image)
    
    except Exception as e:
        return f"Error: {str(e)}"

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Leffa Virtual Try-On")
    gr.Markdown("Upload a person image and a garment image to see the virtual try-on result.")
    
    with gr.Row():
        with gr.Column():
            src_image = gr.Image(type="pil", label="Person Image")
            ref_image = gr.Image(type="pil", label="Garment Image")
        
        with gr.Column():
            guidance_scale = gr.Slider(1.0, 10.0, value=2.5, label="Guidance Scale")
            num_steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
            vt_garment_type = gr.Dropdown(
                choices=["upper_body", "lower_body", "full_body"],
                value="upper_body",
                label="Garment Type"
            )
            seed = gr.Number(value=42, label="Random Seed")
            ref_acceleration = gr.Checkbox(label="Reference Acceleration", value=False)
            vt_repaint = gr.Checkbox(label="Repaint Mode", value=False)
            preprocess_garment = gr.Checkbox(label="Preprocess Garment", value=False)
    
    submit_btn = gr.Button("Generate")
    output_image = gr.Image(label="Generated Image")
    
    submit_btn.click(
        fn=run_virtual_tryon,
        inputs=[
            src_image,
            ref_image,
            guidance_scale,
            num_steps,
            vt_garment_type,
            seed,
            ref_acceleration,
            vt_repaint,
            preprocess_garment
        ],
        outputs=output_image
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False)